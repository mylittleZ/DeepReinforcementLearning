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

# Value Iteration
def value_iteration(gamma=0.9, theta=1e-6):
    values = np.zeros(grid_shape)
    iteration = 0

    while True:
        iteration += 1
        delta = 0


        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                v = values[i, j]
                q_values = []
                for action in actions:
                    (next_i, next_j), reward = next_state_reward((i, j), action)
                    q_values.append(reward + gamma * values[next_i, next_j])
                values[i, j] = max(q_values)
                delta = max(delta, abs(v - values[i, j]))
        if iteration <= 2:
            print(f"\nState values after iteration {iteration}:")
            print(values)
        if delta < theta:
            break

    return values

# Run value iteration
final_values = value_iteration()

print("\nFinal State Values:")
print(final_values)
