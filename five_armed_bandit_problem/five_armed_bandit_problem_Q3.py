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

def run_bandit_algorithms(epsilon, supervised=False):
    # Initialize an array rewards to store rewards at each time step
    rewards = np.zeros(num_steps)
    # Initialize an array to estimate the value of each action
    value_estimate = np.zeros(num_arms)
    # Initialize an array to count the number of times each action is selected.
    action_selection_counts = np.zeros(num_arms)

    # loop to update action,reward,action_selection_counts,value_estimate,rewards
    for step in range(num_steps):
        # supervised learning first 500 steps
        if supervised and step < 500:
            action = step // 100
        else:
            action = select_action(value_estimate, epsilon)
        reward = np.random.normal(q_star[action], 1)
        action_selection_counts[action] += 1
        value_estimate[action] += (reward - value_estimate[action]) / action_selection_counts[action]
        rewards[step] = reward
    return rewards
# store the average reward curves for each policy
average_rewards = np.zeros((4, num_steps))

# Run the bandit problem for each epsilon value and the supervised approach
for run in range(num_runs):
    for i, epsilon in enumerate(epsilon_list):
        average_rewards[i] += run_bandit_algorithms(epsilon)
    average_rewards[-1] += run_bandit_algorithms(0, supervised=True)
average_rewards /= num_runs

# Plot the results
labels = ['greedy', 'ɛ-greedy with ɛ=0.1', 'ɛ-greedy with ɛ=0.05', 'Supervised learning']
for rewards, label in zip(average_rewards, labels):
    plt.plot(rewards, label=label)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()