import random

# Number of arms (actions)
k = 5

# Initialize Q(a) and N(a)
Q = [0.0] * k
N = [0] * k

# Exploration probability
epsilon = 0.1

# Function to simulate the bandit and return a reward
def bandit(action):
    # In this example, each arm provides a random reward from a normal distribution
    # with a different mean and standard deviation.
    means = [1.0, 2.0, 1.5, 3.0, 2.5]
    stddevs = [1.0, 1.0, 1.0, 1.0, 1.0]
    return random.normalvariate(means[action], stddevs[action])

# Number of time steps
num_steps = 100

# Lists to store results
average_rewards = []

# Main loop
for t in range(1, num_steps + 1):
    # Choose action A
    if random.random() > epsilon:
        max_Q = max(Q)
        A = random.choice([a for a, q in enumerate(Q) if q == max_Q])
    else:
        A = random.randint(0, k - 1)
    
    # Simulate the bandit and get the reward R
    R = bandit(A)
    
    # Update the count of action A
    N[A] += 1
    
    # Update the action value estimate Q(A)
    Q[A] += (1 / N[A]) * (R - Q[A])
    
    # Track the average reward over time
    average_rewards.append(sum(Q) / k)

# Print the results
for i, avg_reward in enumerate(average_rewards):
    print(f"Step {i + 1}: Average Reward = {avg_reward:.4f}")
