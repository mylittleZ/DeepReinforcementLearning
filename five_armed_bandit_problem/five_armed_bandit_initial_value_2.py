import numpy as np
import matplotlib.pyplot as plt

# 5-armed bandit problem changed the average reward 𝑞∗(𝑎) after 500 steps

q_star_initial = np.array([0.1, -0.7, 0.8, 0.3, 0.5])
q_star_after_500_steps = np.array([0.1, 0.7, -0.4, 0.3, 0.5])
num_arms = 5
num_steps = 1000
num_runs = 1000

def select_action(value_estimate, epsilon):
    # epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(num_arms)
    # 1-epsilon
    else:
        return np.argmax(value_estimate)

def run_bandit_algorithms(q_star, epsilon, alpha):
    rewards = np.zeros(num_steps)
    value_estimate = np.ones(num_arms) * 5.0
    action_selection_counts = np.zeros(num_arms)

    # loop to update action,reward,action_selection_counts,value_estimate,rewards
    for step in range(num_steps):
        if step == 501:
            q_star = q_star_after_500_steps
        action = select_action(value_estimate, epsilon)
        reward = np.random.normal(q_star[action], 1)
        action_selection_counts[action] += 1
        if alpha == 0:
            value_estimate[action] += (reward - value_estimate[action]) / action_selection_counts[action]
        else:
            value_estimate[action] += (reward - value_estimate[action]) * alpha
        rewards[step] = reward
    return rewards


average_rewards = np.zeros((3, num_steps))

# Run the bandit problem for Greedy and ε-greedy algorithms with different parameter
for run in range(num_runs):
    average_rewards[0] += run_bandit_algorithms(q_star_initial, epsilon=0, alpha=0)
    average_rewards[1] += run_bandit_algorithms(q_star_initial, epsilon=0.1, alpha=0)
    average_rewards[2] += run_bandit_algorithms(q_star_initial, epsilon=0.1, alpha=0.2)

average_rewards /= num_runs

# Plot the results
labels = ['greedy', 'ɛ=0.1 and sample average value estimation;', 'ɛ=0.1 and constant step size alpha=0.2']
for rewards, label in zip(average_rewards, labels):
    plt.plot(rewards, label=label)
plt.xlabel('Steps')
plt.ylabel('Average Reward with initial value estimate of 5')
plt.legend()
plt.show()
