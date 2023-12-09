import numpy as np
import matplotlib.pyplot as plt

class RecyclingRobot:
    def __init__(self):
        self.battery = 100

    def step(self, action):
        if action == 'search':
            reward = np.random.choice([3, 4, 5, 6])
            energy_consumption = np.random.uniform(20, 40)
        elif action == 'wait':
            reward = np.random.choice([0, 1, 2])
            energy_consumption = np.random.uniform(5, 25)
        elif action == 'charge':
            reward = 0
            self.battery = 100
            energy_consumption = 0

        self.battery = max(0, min(100, self.battery - energy_consumption))

        if self.battery == 0:
            reward = -3

        return self.battery, reward

def normalize_state(state):
    return state / 100

def linear_approximation(battery_level, weights):
    normalized_battery_level = normalize_state(battery_level)
    return weights[0] + weights[1] * normalized_battery_level

def update_weights(weights, target, prediction, battery_level, alpha):
    normalized_battery_level = normalize_state(battery_level)
    gradient = np.array([1, normalized_battery_level])
    update = alpha * (target - prediction) * gradient
    update = np.clip(update, -1.0, 1.0)  # Limit the update
    weights += update
    return weights

def select_action(epsilon, weights, battery_level):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [linear_approximation(battery_level, weights[action]) for action in actions]
        return actions[np.argmax(q_values)]

# Parameters
gamma = 0.8
alpha = 0.05  # Learning rate
epsilon = 0.1  # Epsilon for epsilon-greedy strategy
num_episodes = 3000
actions = ['search', 'wait', 'charge']

# Initialize weights for each action
weights = {action: np.random.rand(2) for action in actions}

# Training loop
weights_history = {action: [] for action in actions}
for episode in range(num_episodes):
    robot = RecyclingRobot()
    done = False
    while not done:
        current_battery = robot.battery
        action = select_action(epsilon, weights, current_battery)
        next_battery, reward = robot.step(action)
        done = next_battery == 0 or action == 'charge'
        # TD(0) Update
        target = reward
        if not done:
            next_action = select_action(epsilon, weights, next_battery)
            target += gamma * linear_approximation(next_battery, weights[next_action])
        prediction = linear_approximation(current_battery, weights[action])
        weights[action] = update_weights(weights[action], target, prediction, current_battery, alpha)
        # Record weights for plotting
        for action in actions:
            weights_history[action].append(weights[action].copy())

# Plotting
plt.figure(figsize=(12, 8))
for action in actions:
    weights_data = np.array(weights_history[action])
    plt.plot(weights_data[:, 1], label=f'{action} - w1')
    plt.plot(weights_data[:, 0], label=f'{action} - w2')
plt.xlabel('Episodes')
plt.ylabel('Weights')
plt.title('Estimated Weights During Learning')
plt.legend()
plt.show()
plt.figure(figsize=(12, 8))
for action in actions:
    Q_values = [linear_approximation(battery, weights[action]) for battery in range(101)]
    plt.plot(range(101), Q_values, label=f'Q-values for {action}')
plt.xlabel('Battery Level')
plt.ylabel('Q-values')
plt.title('Learned Action Values for all Possible States')
plt.legend()
plt.show()
