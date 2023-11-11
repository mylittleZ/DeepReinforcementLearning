# For the following simplified grid world, assuming that each on-grid
# transition leads to a reward of -1, all off-grid transitions lead to a reward of -
# 10 with no state change, and discount factor of 1, calculate and compare the
# state values using Bellman equation for following policies:
# a. 𝜋(a|s) = 0.25, for a = left, right, up, and down;
# b. 𝜋(a|s) = 0.5, for a = left and up; 𝜋(a|s) = 0, for a = right and down.

import numpy as np

# Dimensions of the grid
width, height = 3, 3

# Rewards
on_grid_reward = -1
off_grid_reward = -10

# Discount factor
gamma = 1

# Terminal state
terminal_state = (0, 0)

# Action outcomes
actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # up, left, down, right

# Convergence criterion
epsilon = 0.001

# Compute for policy a
pi_left_up_a = 0.25
pi_right_down_a = 0.25
def compute_values(pi_left_up, pi_right_down):
    # Initialize state values.
    V = np.zeros((width, height))

    # Start the iteration to converge to the true state values
    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for i in range(width):
            for j in range(height):
                if (i, j) == terminal_state:
                    continue
                old_value = V[i, j]
                value = 0
                for idx, action in enumerate(actions):
                    next_i, next_j = i + action[0], j + action[1]

                    # Get action probability based on the policy
                    if idx < 2:  # left or up
                        action_prob = pi_left_up
                    else:  # right or down
                        action_prob = pi_right_down

                    # Check if the resulting state is off the grid
                    if next_i < 0 or next_i >= width or next_j < 0 or next_j >= height:
                        reward = off_grid_reward
                        value += action_prob * (reward + gamma * V[i, j])
                    else:
                        reward = on_grid_reward
                        value += action_prob * (reward + gamma * V[next_i, next_j])

                V[i, j] = value
                delta = max(delta, abs(old_value - V[i, j]))

    return V



values_a = compute_values(pi_left_up_a, pi_right_down_a)
print("state values for Policy A:")
print(values_a)

# Compute for policy b
pi_left_up_b = 0.5
pi_right_down_b = 0.0
values_b = compute_values(pi_left_up_b, pi_right_down_b)
print("\nstate values for Policy B:")
print(values_b)
