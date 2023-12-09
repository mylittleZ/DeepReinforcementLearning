import numpy as np
import matplotlib.pyplot as plt


ALPHA = 0.25
BETA = 0.25
GAMMA = 1
NUM_EPISODES = 1000
NUM_STEPS = 4


SEARCH_REWARDS = [3, 4, 5, 6]
WAIT_REWARDS = [0, 1, 2]
RECHARGE_REWARD = 0
DEPLETING_REWARD = -3


Q = {
    'high': {'search': 0, 'wait': 0},
    'low': {'search': 0, 'wait': 0, 'recharge': 0}
}
returns = {state: {action: [] for action in actions} for state, actions in Q.items()}

policy = {
    'high': np.random.choice(['search', 'wait']),
    'low': np.random.choice(['search', 'wait', 'recharge'])
}

Q_values = {state: {action: [] for action in actions} for state, actions in Q.items()}


def step(state, action):
    if state == 'high':
        if action == 'wait':
            return state, np.random.choice(WAIT_REWARDS)
        elif action == 'search':
            return ('high' if np.random.rand() < ALPHA else 'low', np.random.choice(SEARCH_REWARDS))
    elif state == 'low':
        if action == 'recharge':
            return 'high', RECHARGE_REWARD
        elif action == 'wait':
            return state, np.random.choice(WAIT_REWARDS)
        elif action == 'search':
            if np.random.rand() < BETA:
                return state, np.random.choice(SEARCH_REWARDS)
            else:
                return 'high', DEPLETING_REWARD
    return state, 0

for episode in range(NUM_EPISODES):
    state = np.random.choice(['high', 'low'])
    action = np.random.choice(list(Q[state].keys()))


    episode = []
    for _ in range(NUM_STEPS):
        next_state, reward = step(state, action)
        episode.append((state, action, reward))
        state = next_state
        action = policy[state]

    G = 0
    for t in range(len(episode)-1, -1, -1):
        state, action, reward = episode[t]
        G = GAMMA * G + reward
        if not any(x[0] == state and x[1] == action for x in episode[:t]):
            returns[state][action].append(G)
            Q[state][action] = np.mean(returns[state][action])
            A_star = max(Q[state], key=Q[state].get)
            policy[state] = A_star

    for state in Q_values:
        for action in Q_values[state]:
            Q_values[state][action].append(Q[state][action])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
for ax, (state, actions) in zip(axes.flatten(), Q_values.items()):
    for action, values in actions.items():
        ax.plot(values, label=f"{action} ({state})")
    ax.set_title(f"First-visit MC with exploring start estimated action values for State: '{state}'")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Estimated Q-value')
    ax.legend()

plt.tight_layout()
plt.show()