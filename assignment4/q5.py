import numpy as np
import matplotlib.pyplot as plt

# Constants for Q-learning
epsilon_ql = 0.1
alpha_ql = 0.05  # Step size for Q-learning update
gamma_ql = 0.8
steps_ql = 5000
alpha = 0.25
beta = 0.25

# Rewards
search_rewards = [3, 4, 5, 6]
wait_rewards = [0, 1, 2]
recharge_reward = 0
deplete_reward = -3


Q_ql = {
    'high': {'search': 0, 'wait': 0},
    'low': {'search': 0, 'wait': 0, 'recharge': 0}
}

Q_history_ql = {
    'high': {'search': [], 'wait': []},
    'low': {'search': [], 'wait': [], 'recharge': []}
}


# select an action using epsilon-greedy policy
def choose_action_epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(list(Q[state].keys()))
    else:
        return max(Q[state], key=Q[state].get)


# simulate the environment step based on the current state and action
def simulate_step(state, action, alpha, beta):
    if state == 'high':
        if action == 'wait':
            return 'high', np.random.choice(wait_rewards)
        elif action == 'search':
            if np.random.rand() < alpha:
                return 'high', np.random.choice(search_rewards)
            else:
                return 'low', np.random.choice(search_rewards)
    else:
        if action == 'recharge':
            return 'high', recharge_reward
        elif action == 'wait':
            return 'low', np.random.choice(wait_rewards)
        elif action == 'search':
            if np.random.rand() < beta:
                return 'low', np.random.choice(search_rewards)
            else:
                return 'high', deplete_reward


# Q-learning learning loop
state = np.random.choice(['high', 'low'])

for step in range(steps_ql):
    # Choose action using policy derived from Q (epsilon-greedy)
    action = choose_action_epsilon_greedy(state, Q_ql, epsilon_ql)
    next_state, reward = simulate_step(state, action, alpha, beta)

    # Q-learning update
    best_next_action = max(Q_ql[next_state], key=Q_ql[next_state].get)
    Q_ql[state][action] += alpha_ql * (reward + gamma_ql * Q_ql[next_state][best_next_action] - Q_ql[state][action])

    # Record action values for plotting
    for s in Q_history_ql:
        for a in Q_history_ql[s]:
            Q_history_ql[s][a].append(Q_ql[s][a])

    state = next_state

# Plot the estimated action values for high state
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for action in Q_history_ql['high']:
    plt.plot(Q_history_ql['high'][action], label=f"high, {action}")
plt.title('Estimated Action Values for High State (Q-learning)')
plt.xlabel('Steps')
plt.ylabel('Estimated Action Value')
plt.legend()

# Plot the estimated action values for low state
plt.subplot(1, 2, 2)
for action in Q_history_ql['low']:
    plt.plot(Q_history_ql['low'][action], label=f"low, {action}")
plt.title('Estimated Action Values for Low State (Q-learning)')
plt.xlabel('Steps')
plt.ylabel('Estimated Action Value')
plt.legend()

plt.tight_layout()
plt.show()
