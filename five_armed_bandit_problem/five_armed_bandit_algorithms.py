import numpy as np
import matplotlib.pyplot as plt

# 5-armed bandit problem

q_star = np.array([0.1, -0.7, 0.8, 0.3, 0.5])
num_arms = len(q_star)
num_steps = 1000
num_runs = 1000
epsilon_values = [0, 0.1, 0.05]

# Supervised learning approach parameters
supervised_steps = 500
supervised_freq = 100

def select_action(value_estimate, epsilon):
    # epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(num_arms)
    # 1-epsilon
    else:
        return np.argmax(value_estimate)

def bandit_run(epsilon, supervised=False):
    # Initialize an array rewards to store rewards at each time step
    rewards = np.zeros(num_steps)
    # Initialize an array to estimate the value of each action
    # value_estimate = np.zeros(num_arms)
    value_estimate = np.ones(num_arms) * 5.0
    # Initialize an array to count the number of times each action is selected.
    action_count = np.zeros(num_arms)

    for step in range(num_steps):
        # supervised learning first 500 steps
        if supervised and step < supervised_steps:
            action = step // supervised_freq
        else:
            action = select_action(value_estimate, epsilon)
        # update reward,action_count,value_estimate,rewards
        reward = np.random.normal(q_star[action], 1)
        action_count[action] += 1
        value_estimate[action] += (reward - value_estimate[action]) / action_count[action]
        rewards[step] = reward
    return rewards
# store the average reward curves for each policy
average_rewards = np.zeros((len(epsilon_values) + 1, num_steps))

# Run the bandit problem for each epsilon value and the supervised approach
for run in range(num_runs):
    for i, epsilon in enumerate(epsilon_values):
        average_rewards[i] += bandit_run(epsilon)
    average_rewards[-1] += bandit_run(0, supervised=True)
average_rewards /= num_runs

# Plot the results
labels = ['Greedy', 'Epsilon=0.1', 'Epsilon=0.05', 'Supervised']
for rewards, label in zip(average_rewards, labels):
    plt.plot(rewards, label=label)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()