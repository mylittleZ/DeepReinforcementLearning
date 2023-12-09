import numpy as np
import matplotlib.pyplot as plt

# Constants for the one-step TD learning
alpha_td = 0.05  # Step size
gamma_td = 0.8
steps = 4000
alpha = 0.25  # Probability of staying in the same state when action is 'search' in high state
beta = 0.25  # Probability of staying in the same state when action is 'search' in low state

# Rewards
search_rewards = [3, 4, 5, 6]
wait_rewards = [0, 1, 2]
recharge_reward = 0
deplete_reward = -3

policy = {
    'high': {'search': 0.5, 'wait': 0.5},
    'low': {'search': 0.5, 'wait': 0.25, 'recharge': 0.25}
}

V = {'high': 0, 'low': 0}

# Store estimated state values for plotting
V_history = {'high': [], 'low': []}


# simulate the environment step based on the current state and action
def simulate_step(state, action):
    if state == 'high':
        if action == 'wait':
            return 'high', np.random.choice(wait_rewards)
        elif action == 'search':
            if np.random.rand() < alpha:
                return 'high', np.random.choice(search_rewards)
            else:
                return 'low', np.random.choice(search_rewards)
    else:  # state is 'low'
        if action == 'recharge':
            return 'high', recharge_reward
        elif action == 'wait':
            return 'low', np.random.choice(wait_rewards)
        elif action == 'search':
            if np.random.rand() < beta:
                return 'low', np.random.choice(search_rewards)
            else:
                return 'high', deplete_reward


# choose an action according to the stochastic policy
def choose_action_stochastic(state, policy):
    actions, probabilities = zip(*policy[state].items())
    return np.random.choice(actions, p=probabilities)


# One-step TD learning loop
state = np.random.choice(['high', 'low'])
for step in range(steps):
    action = choose_action_stochastic(state, policy)
    next_state, reward = simulate_step(state, action)
    V[state] = V[state] + alpha_td * (reward + gamma_td * V[next_state] - V[state])

    # Record state values for plotting
    V_history['high'].append(V['high'])
    V_history['low'].append(V['low'])
    state = next_state

plt.figure(figsize=(12, 6))
plt.plot(V_history['high'], label='High State Value')
plt.plot(V_history['low'], label='Low State Value')
plt.xlabel('Steps')
plt.ylabel('Estimated State Value')
plt.title('Estimated State Values for One-Step TD Learning')
plt.legend()
plt.show()
