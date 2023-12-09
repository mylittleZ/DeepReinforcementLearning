import numpy as np
import matplotlib.pyplot as plt

# Constants for SARSA
epsilon = 0.1
alpha_sarsa = 0.05
gamma_sarsa = 0.8
steps_sarsa = 5000
alpha = 0.25
beta = 0.25

# Rewards
search_rewards = [3, 4, 5, 6]
wait_rewards = [0, 1, 2]
recharge_reward = 0
deplete_reward = -3

Q_sarsa = {
    'high': {'search': 0, 'wait': 0},
    'low': {'search': 0, 'wait': 0, 'recharge': 0}
}

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


# select an action using epsilon-greedy policy
def choose_action_epsilon_greedy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(list(Q[state].keys()))
    else:
        # Exploit: best action based on current Q
        return max(Q[state], key=Q[state].get)


# SARSA learning loop
state = np.random.choice(['high', 'low'])
action = choose_action_epsilon_greedy(state, Q_sarsa, epsilon)

Q_history_sarsa = {
    'high': {'search': [], 'wait': []},
    'low': {'search': [], 'wait': [], 'recharge': []}
}

for step in range(steps_sarsa):
    # Take action and observe reward and next state
    next_state, reward = simulate_step(state, action, alpha, beta)
    # Choose next action using epsilon-greedy
    next_action = choose_action_epsilon_greedy(next_state, Q_sarsa, epsilon)

    # SARSA update
    Q_sarsa[state][action] += alpha_sarsa * (
                reward + gamma_sarsa * Q_sarsa[next_state][next_action] - Q_sarsa[state][action])

    for s in Q_history_sarsa:
        for a in Q_history_sarsa[s]:
            Q_history_sarsa[s][a].append(Q_sarsa[s][a])

    # Transition to the next state and action
    state, action = next_state, next_action

# Plot the estimated action values for high state
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
for action in Q_history_sarsa['high']:
    plt.plot(Q_history_sarsa['high'][action], label=f"high, {action}")
plt.title('Estimated Action Values for High State (SARSA)')
plt.xlabel('Steps')
plt.ylabel('Estimated Action Value')
plt.legend()

# Plot the estimated action values for low state
plt.subplot(1, 2, 2)
for action in Q_history_sarsa['low']:
    plt.plot(Q_history_sarsa['low'][action], label=f"low, {action}")
plt.title('Estimated Action Values for Low State (SARSA)')
plt.xlabel('Steps')
plt.ylabel('Estimated Action Value')
plt.legend()

plt.tight_layout()
plt.show()
