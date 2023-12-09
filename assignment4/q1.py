import numpy as np
import matplotlib.pyplot as plt

# initialization
alpha = 0.25
beta = 0.25
gamma = 1.0
num_episodes = 1000
num_steps = 4

states = ['high', 'low']
actions = ['search', 'wait', 'recharge']

# Rewards
rewards_search = [3, 4, 5, 6]
rewards_wait = [0, 1, 2]
reward_recharge = 0
reward_deplete = -3

policy = {
    'high': {'search': 0.5, 'wait': 0.5},
    'low': {'search': 0.5, 'wait': 0.25, 'recharge': 0.25}
}

# choose an action based on the stochastic policy
def choose_action(state):
    return np.random.choice(a=list(policy[state].keys()), p=list(policy[state].values()))

# get reward based on action and state transition
def get_reward(action, current_state, next_state):
    if action == 'search':
        if current_state == 'high':
            return np.random.choice(rewards_search)
        elif current_state == 'low' and next_state == 'high':
            return reward_deplete
    elif action == 'wait':
        return np.random.choice(rewards_wait)
    return 0

# Initialize state-value estimates and returns
V = {state: 0.0 for state in states}
returns = {state: [] for state in states}
V_over_time = {'high': [], 'low': []}

# Monte Carlo simulation with tracking of state values over episodes
for episode in range(num_episodes):
    episode_states, episode_rewards = [], []
    current_state = np.random.choice(states)

    for step in range(num_steps):
        action = choose_action(current_state)
        next_state = 'high' if action == 'recharge' else current_state

        if action == 'search':
            if current_state == 'high':
                next_state = 'high' if np.random.rand() < alpha else 'low'
            elif current_state == 'low':
                next_state = 'low' if np.random.rand() < beta else 'high'

        reward = get_reward(action, current_state, next_state)
        episode_states.append(current_state)
        episode_rewards.append(reward)
        # Update the current state
        current_state = next_state

    # First-visit Monte Carlo prediction
    G = 0
    for t in range(len(episode_states)-1, -1, -1):
        G = gamma * G + episode_rewards[t]
        if episode_states[t] not in episode_states[:t]:
            returns[episode_states[t]].append(G)
            V[episode_states[t]] = np.mean(returns[episode_states[t]])

    # Track the state values over time for plotting
    V_over_time['high'].append(V['high'])
    V_over_time['low'].append(V['low'])

plt.figure(figsize=(10, 5))
plt.plot(V_over_time['high'], label='Estimated Value of High State')
plt.plot(V_over_time['low'], label='Estimated Value of Low State')
plt.xlabel('Episodes')
plt.ylabel('Estimated State Value')
plt.title('Estimated State Values for First-visit MC')
plt.legend()
plt.show()
