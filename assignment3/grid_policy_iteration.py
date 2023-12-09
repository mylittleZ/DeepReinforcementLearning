import numpy as np

# Define the environment
grid_shape = (5, 5)

A, A_prime = (0, 1), (4, 1)
B, B_prime = (0, 3), (2, 3)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
# Function to find the next state and reward
def next_state_reward(s, a):
    if s == A:
        return A_prime, 10
    if s == B:
        return B_prime, 5

    next_s = (s[0] + a[0], s[1] + a[1])

    if 0 <= next_s[0] < grid_shape[0] and 0 <= next_s[1] < grid_shape[1]:
        return next_s, 0
    return s, -1

def policy_iteration(gamma=0.9, theta=1e-6):
    # Random policy
    policy = np.ones((grid_shape[0], grid_shape[1], len(actions))) / len(actions)
    values = np.zeros(grid_shape)

    intermediate_policies = []
    intermediate_values = []

    iteration_count = 0

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    v = values[i, j]
                    new_v = 0
                    for action in actions:
                        (next_i, next_j), reward = next_state_reward((i, j), action)
                        new_v += policy[i, j, actions.index(action)] * (reward + gamma * values[next_i, next_j])
                    values[i, j] = new_v
                    delta = max(delta, abs(v - values[i, j]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                old_action_probs = policy[i, j].copy()
                q_values = [0] * len(actions)
                for k, action in enumerate(actions):
                    (next_i, next_j), reward = next_state_reward((i, j), action)
                    q_values[k] = reward + gamma * values[next_i, next_j]
                best_action = np.argmax(q_values)
                policy[i, j] = np.eye(len(actions))[best_action]
                if not np.array_equal(policy[i, j], old_action_probs):
                    policy_stable = False

        iteration_count += 1
        if iteration_count == 1 or iteration_count == 2:
            intermediate_policies.append(policy.copy())
            intermediate_values.append(values.copy())
        if policy_stable:
            return policy, values, intermediate_policies, intermediate_values

optimal_policy, optimal_values, inter_policies, inter_values = policy_iteration()

print("First state values:")
print(inter_values[0])
print("\nFirst Policy (0=up, 1=down, 2=left, 3=right):")
print(np.argmax(inter_policies[0], axis=2))

print("\nSecond state values:")
print(inter_values[1])
print("\nSecond Policy (0=up, 1=down, 2=left, 3=right):")
print(np.argmax(inter_policies[1], axis=2))

print("\nOptimal state values:")
print(optimal_values)
print("\nOptimal Policy (0=up, 1=down, 2=left, 3=right):")
print(np.argmax(optimal_policy, axis=2))