import numpy as np
import matplotlib.pyplot as plt

# Define the parameters and initialize the state values
alpha = 0.25  # state transition probability
beta = 0.25  # state transition probability
gamma = 0.8  # discount factor
step_size = 0.05  # step-size parameter for TD update
num_steps = 4000  # number of steps for the TD algorithm

# Rewards
search_rewards = [3, 4, 5, 6]
wait_rewards = [0, 1, 2]
recharge_reward = 0
deplete_reward = -3

# Initialize state values
V = {'high': 0, 'low': 0}

# Define the policy
policy = {
    'high': {'search': 0.5, 'wait': 0.5},
    'low': {'search': 0.5, 'wait': 0.25, 'recharge': 0.25}
}


# Choose action based on policy
def choose_action(state):
    actions = list(policy[state].keys())
    probabilities = list(policy[state].values())
    return np.random.choice(actions, p=probabilities)


# TD update function
def td_update(state, reward, next_state):
    V[state] += step_size * (reward + gamma * V[next_state] - V[state])


# Simulate the steps
high_values, low_values = [], []
for _ in range(num_steps):
    current_state = np.random.choice(['high', 'low'])
    action = choose_action(current_state)

    # Determine the reward and next state
    if current_state == 'high':
        if action == 'search':
            reward = np.random.choice(search_rewards)
            next_state = 'high' if np.random.rand() < alpha else 'low'
        else:  # action == 'wait'
            reward = np.random.choice(wait_rewards)
            next_state = 'high'
    else:  # current_state == 'low'
        if action == 'search':
            reward = deplete_reward
            next_state = 'low' if np.random.rand() < beta else 'high'
        elif action == 'wait':
            reward = np.random.choice(wait_rewards)
            next_state = 'low'
        else:  # action == 'recharge'
            reward = recharge_reward
            next_state = 'high'

    # Perform the TD update
    td_update(current_state, reward, next_state)

    # Record the state values for plotting
    high_values.append(V['high'])
    low_values.append(V['low'])

# Plot the estimated state values
plt.figure(figsize=(12, 6))
plt.plot(high_values, label='High State Value')
plt.plot(low_values, label='Low State Value')
plt.xlabel('Steps')
plt.ylabel('Estimated State Value')
plt.title('Estimated State Values over 4000 Steps with One-Step TD')
plt.legend()
plt.show()

