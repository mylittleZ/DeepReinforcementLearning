import numpy as np
import matplotlib.pyplot as plt

q_star = np.array([0.1, -0.7, 0.8, 0.3, 0.5])
num_arms = 5
num_steps = 1000
num_runs = 1000
epsilon_list = [0, 0.1, 0.05]

def select_action(value_estimate, epsilon):
    # epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(num_arms)
    # 1-epsilon
    else:
        return np.argmax(value_estimate)

def run_bandit_algorithms(epsilon):
    # Initialize an array rewards to store rewards at each time step
    rewards = np.zeros(num_steps)
    # Initialize an array to estimate the value of each action
    value_estimate = np.ones(num_arms) * 5.0
    # Initialize an array to count the number of times each action is selected.
    action_selection_counts = np.zeros(num_arms)

    # loop to update action,reward,action_selection_counts,value_estimate,rewards
    for step in range(num_steps):
        action = select_action(value_estimate, epsilon)
        reward = np.random.normal(q_star[action], 1)
        action_selection_counts[action] += 1
        value_estimate[action] += (reward - value_estimate[action]) / action_selection_counts[action]
        rewards[step] = reward
    return rewards
# store the average reward curves for each policy
average_rewards = np.zeros((3, num_steps))

# Run the bandit problem for each epsilon value
for run in range(num_runs):
    for i, epsilon in enumerate(epsilon_list):
        average_rewards[i] += run_bandit_algorithms(epsilon)
average_rewards /= num_runs

# Plot the results
labels = ['greedy', 'ɛ-greedy with ɛ=0.1', 'ɛ-greedy with ɛ=0.05']
for rewards, label in zip(average_rewards, labels):
    plt.plot(rewards, label=label)
plt.xlabel('Steps')
plt.ylabel('Average Reward with initial value estimate of 5')
plt.legend()
plt.show()