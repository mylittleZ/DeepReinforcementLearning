import numpy as np
import matplotlib.pyplot as plt

class RecyclingRobot:
    def __init__(self):
        self.battery_level = 100

    def step(self, action):
        done = False
        reward = 0

        if action == 'Search':
            reward = np.random.choice([3, 4, 5, 6])
            energy_consumed = np.random.uniform(20, 40)
        elif action == 'Wait':
            reward = np.random.choice([0, 1, 2])
            energy_consumed = np.random.uniform(5, 25)
        else:  # Charge
            energy_consumed = - (self.battery_level - 100)  # Recharge to 100
            reward = 0  # No reward for charging

        self.battery_level -= energy_consumed

        if self.battery_level <= 0:
            reward = -3  # Penalty for battery depletion
            self.battery_level = 0
            done = True

        return self.battery_level, reward, done

# Initialize weights for linear function approximation
weights = {
    'Search': np.random.rand(2),
    'Wait': np.random.rand(2),
    'Charge': np.random.rand(2)
}

# Normalize state function
def normalize_state(state):
    return state / 100

# Q-value function approximation
def q_value_function(state, weights):
    normalized_state = normalize_state(state)
    return weights[0] + weights[1] * normalized_state

# Update weights using Stochastic Gradient Descent (SGD)
def update_weights(weights, target, prediction, state, alpha=0.05):
    normalized_state = normalize_state(state)
    gradient = np.array([1, normalized_state])
    weights += alpha * (target - prediction) * gradient
    return weights

# Choose an action based on an epsilon-greedy strategy
def choose_action(battery, weights, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(['Search', 'Wait', 'Charge'])
    else:
        q_values = {action: q_value_function(battery, weights[action]) for action in ['Search', 'Wait', 'Charge']}
        return max(q_values, key=q_values.get)

# Initialize variables for training
num_episodes = 5000  # Reduced number of episodes for faster execution
gamma = 0.8  # Discount factor
alpha = 0.05  # Learning rate

weights_history = {action: [] for action in ['Search', 'Wait', 'Charge']}

# Training loop
for episode in range(num_episodes):
    robot = RecyclingRobot()
    done = False

    while not done:
        current_state = robot.battery_level
        current_action = choose_action(current_state, weights)
        next_state, reward, done = robot.step(current_action)

        # Compute the target and update weights
        if not done:
            next_action = choose_action(next_state, weights)
            target = reward + gamma * q_value_function(next_state, weights[next_action])
        else:
            target = reward

        prediction = q_value_function(current_state, weights[current_action])
        weights[current_action] = update_weights(weights[current_action], target, prediction, current_state)

        # Record weights history for plotting
        for action in weights:
            weights_history[action].append(weights[action].copy())

# Plotting the weights history
plt.figure(figsize=(12, 6))
for action in weights_history:
    w0 = [weight[0] for weight in weights_history[action]]
    w1 = [weight[1] for weight in weights_history[action]]
    plt.plot(w0, label=f'{action} w0')
    plt.plot(w1, label=f'{action} w1')
plt.xlabel('Episodes')
plt.ylabel('Weights')
plt.legend()
plt.title('Estimated Weights During Learning')
plt.show()

# Plotting the learned action values for all possible states
plt.figure(figsize=(12, 6))
states = np.arange(101)
for action in weights:
    q_values = [q_value_function(state, weights[action]) for state in states]
    plt.plot(states, q_values, label=f'Q-values for {action}')
plt.xlabel('Battery Level')
plt.ylabel('Q-values')
plt.legend()
plt.title('Learned Action Values for all Possible States')
plt.show()
