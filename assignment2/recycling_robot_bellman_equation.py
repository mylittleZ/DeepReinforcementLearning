# Using Bellman equation to calculate and compare the action values for the
# recycling robot example with 𝛼 = 0.25; 𝛽 = 0.25; 𝑟search = 5; 𝑟wait = 1;
# 𝛾 = 0.8 under the following policies:
# a. 𝜋(high) = search; 𝜋(low) = recharge;
# b. 𝜋(search|high) = 0.5; 𝜋(wait|high) = 0.5; 𝜋(search|low) = 0.25;
# 𝜋(wait|low) = 0.5; 𝜋(recharge|low) = 0.25.
# Set up the Bellman optimality equation for optimal action values 𝑞∗(𝑠, 𝑎) for
# the recycling robot problem.

# Parameters
alpha = 0.25
beta = 0.25
r_search = 5
r_wait = 1
gamma = 0.8

# Transition probabilities and rewards for state-action pairs
P = {
    'high': {
        'search': {'high': alpha, 'low': 1 - alpha},
        'wait': {'high': 1}
    },
    'low': {
        'search': {'high': 1 - beta, 'low': beta},
        'wait': {'low': 1},
        'recharge': {'high': 1}
    }
}

R = {
    'high': {'search': r_search, 'wait': r_wait},
    'low': {'search': {'high': -3, 'low': r_search}, 'wait': r_wait, 'recharge': 0}
}

def bellman_update(state, action, q_values):
    if state == 'low' and action == 'search':
        high_reward = R[state][action]['high']
        low_reward = R[state][action]['low']
        high_prob = P[state][action]['high']
        low_prob = P[state][action]['low']
        value = high_reward * high_prob + low_reward * low_prob + gamma * (
                    high_prob * max(q_values['high'].values()) +
                    low_prob * max(q_values['low'].values()))
    else:
        value = R[state][action]
        for next_state, prob in P[state][action].items():
            value += gamma * prob * max(q_values[next_state].values())
    return value

def compute_optimal_q_values(max_iterations=1000, threshold=1e-6):
    q_values = {'high': {'search': 0, 'wait': 0}, 'low': {'search': 0, 'wait': 0, 'recharge': 0}}

    for _ in range(max_iterations):
        delta = 0
        new_q_values = q_values.copy()
        for state in ['high', 'low']:
            for action in P[state]:
                new_value = bellman_update(state, action, q_values)
                delta = max(delta, abs(new_value - q_values[state][action]))
                new_q_values[state][action] = new_value
        q_values = new_q_values
        if delta < threshold:
            break

    return q_values

# Compute optimal q-values
optimal_q_values = compute_optimal_q_values()
print(optimal_q_values)
