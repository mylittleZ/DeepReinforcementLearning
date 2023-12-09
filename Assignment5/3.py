import numpy as np
import matplotlib.pyplot as plt

# Constants based on the problem statement
num_episodes = 8000
step_size = 0.05
gamma = 0.8
num_actions = 3
state_space = 101

# Initialize weights
weights = np.zeros((num_actions, 2))

# Reward and energy consumption ranges for each action
rewards = np.array([3, 1, 0])
energy_consumption = [(20, 40), (5, 25), (0, 0)]  # For charge, the consumption is (0,0) as it resets to 100

# Arrays to store weights for plotting
w1_history = np.zeros((num_episodes, num_actions))
w2_history = np.zeros((num_episodes, num_actions))

# Stochastic gradient descent with TD(0)
for episode in range(num_episodes):
    state = np.random.randint(0, state_space)  # Start at a random state
    action = np.random.choice(num_actions)  # Choose a random action
    reward = np.random.choice([rewards[action]])
    energy = np.random.uniform(energy_consumption[action][0], energy_consumption[action][1])

    # Calculate next state based on action
    if action == 2:  # Charge action
        next_state = state_space - 1
    else:
        next_state = max(0, state - energy)  # Ensure state doesn't go below 0

    # Compute TD target and TD error
    if action == 2:  # Charge action
        td_target = reward
    else:
        td_target = reward + gamma * (weights[action, 0] * next_state + weights[action, 1])

    td_error = td_target - (weights[action, 0] * state + weights[action, 1])

    # Update weights
    weights[action, 0] += step_size * td_error * state
    weights[action, 1] += step_size * td_error

    # Store weights for plotting
    w1_history[episode] = weights[:, 0]
    w2_history[episode] = weights[:, 1]

# Plot the estimated weights during learning
plt.figure(figsize=(10, 8))
for i in range(num_actions):
    plt.plot(w1_history[:, i], label=f'w1(action {i})')
    plt.plot(w2_history[:, i], label=f'w2(action {i})')

plt.title('The estimated weights during learning')
plt.xlabel('Episodes')
plt.ylabel('Weight')
plt.legend()
plt.show()

# To plot the learned action values for all possible states, we need to compute the Q values
# for each state-action pair based on the learned weights
q_values = np.zeros((state_space, num_actions))
states = np.arange(state_space)
for action in range(num_actions):
    q_values[:, action] = weights[action, 0] * states + weights[action, 1]

# Plot the learned action values for all possible states
plt.figure(figsize=(10, 8))
for action in range(num_actions):
    plt.plot(states, q_values[:, action], label=f'Action {action}')

plt.title('Learned action values for all possible states')
plt.xlabel('States')
plt.ylabel('Q value')
plt.legend()
plt.show()
