import numpy as np

# Constants and Hyperparameters
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
input_size = 2
hidden_size = 4
output_size = len(ACTIONS)
learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.1
batch_size = 16
memory_size = 10000




# Neural Network

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2

    def train(self, x, target, learning_rate=0.01):
        # Check input shape
        if x.shape[1] != self.W1.shape[0]:
            raise ValueError("Input dimension does not match with network input size")
        if target.shape[1] != self.W2.shape[1]:
            raise ValueError("Target dimension does not match with network output size")

        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2

        # Compute the loss
        loss = np.mean(np.square(target - z2))

        # Backward pass
        delta2 = z2 - target
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(a1)
        dW1 = np.dot(x.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return loss



# FrozenLake Environment
class FrozenLake:
    def __init__(self):
        self.grid = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ]
        self.x, self.y = 0, 0
        self.done = False

    def step(self, action):

        if self.done:
            raise Exception("Game is over!")
        
        if action == UP and self.x > 0:
            self.x -= 1
        elif action == DOWN and self.x < 3:
            self.x += 1
        elif action == LEFT and self.y > 0:
            self.y -= 1
        elif action == RIGHT and self.y < 3:
            self.y += 1
  

        cell = self.grid[self.x][self.y]
        if cell == 'G':
            reward = 1
            self.done = True
        elif cell == 'H':
            reward = -1
            self.done = True
        else:
            reward = 0

        return (self.x, self.y), reward, self.done

    def reset(self):
        self.x, self.y = 0, 0
        self.done = False
        return (self.x, self.y)


# Replay Memory
class ReplayMemory:
    def __init__(self):
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > memory_size:
            del self.memory[0]

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.memory), batch_size)
        return [self.memory[i] for i in indices]

# ... [ Neural Network (SimpleNN) and FrozenLake classes from the previous code ] ...
class DQLAgentWithTarget:
    def __init__(self):
        self.online_nn = SimpleNN(input_size,hidden_size,output_size)
        self.target_nn = SimpleNN(input_size,hidden_size,output_size)
        self.memory = ReplayMemory()
        self.update_target_every = 20  # Hyperparameter: how often to update target network
        self.steps = 0
        self.target_nn.W1, self.target_nn.b1, self.target_nn.W2, self.target_nn.b2 = \
            self.online_nn.W1.copy(), self.online_nn.b1.copy(), self.online_nn.W2.copy(), self.online_nn.b2.copy()

    def act(self, state):
        global epsilon
        if np.random.rand() <= epsilon:
            return np.random.choice(ACTIONS)
        q_values = self.online_nn.forward(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self):
        global epsilon

        if len(self.memory.memory) < batch_size:
            return 0

        batch = self.memory.sample(batch_size)
        batch_loss = 0

        for state, action, reward, next_state, done in batch:
            target = self.online_nn.forward(state)
            if done:
                target[0][action] = reward
            else:
                
                Q_future = max(self.target_nn.forward(next_state)[0])
                
                target[0][action] = reward + gamma * Q_future
            
            loss = self.online_nn.train(state, target)
            batch_loss += loss

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_nn.W1, self.target_nn.b1, self.target_nn.W2, self.target_nn.b2 = \
                self.online_nn.W1.copy(), self.online_nn.b1.copy(), self.online_nn.W2.copy(), self.online_nn.b2.copy()

        return batch_loss / batch_size
   
# DQL Agent with Experience Replay
# ... [ Other parts of the code remain unchanged ] ...

# Main function for Deep Q-Learning with Experience Replay and Q-Fixed Target

def state_representation(state):
    """Converts the state tuple to a numpy array."""
    return np.array([state]).astype(np.float32)

def main():
    # Initialize the agent and environment
    agent = DQLAgentWithTarget()
    env = FrozenLake()
    episodes = 100  # Number of training episodes
    max_steps = 100  # Maximum steps per episode
    track_rewards = []

    # Training the agent

    for episode in range(episodes):
        state = env.reset()
        state = state_representation(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = state_representation(next_state)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.replay()  # Train the agent
            if done:
                break
            

        track_rewards.append(total_reward)
        if episode % 10 == 0 :
            
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Loss: {loss:.4f}, Epsilon: {epsilon:.4f}")

    
    total_test_episodes = 1
    test_rewards = []

    for episode in range(total_test_episodes):
        state = env.reset()
        state = state_representation(state)
        

        total_reward = 0

        for step in range(max_steps):
            
            action = np.argmax(agent.online_nn.forward(state))
            next_state, reward, done = env.step(action)
            
            print(f"Step {step}: move {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]} / Position: {state} -> {next_state}")
            state = state_representation(next_state)
            total_reward += reward
            

            if done:
                break
            
        test_rewards.append(total_reward)


if __name__ == "__main__":
    main()
