import gymnasium as gym

# --- EXERCISE 2: Performance Evaluation of the Random Agent ---

# Create the FrozenLake environment (non-slippery by default; use is_slippery=False for easier testing)
env = gym.make("FrozenLake-v1", render_mode=None)

# Total number of episodes to run
num_episodes = 1000

# List to keep track of rewards per episode
rewards_per_episode = []

for episode in range(num_episodes):
    # Reset environment at the start of each episode
    observation, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0

    # Run one episode
    while not terminated and not truncated:
        # Choose a random action
        action = env.action_space.sample()
        
        # Take the action
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        episode_reward += reward

        # Update observation (not strictly necessary for random agent)
        observation = next_observation

    # Store the total reward of this episode
    rewards_per_episode.append(episode_reward)

# Calculate the average reward (success rate)
average_reward = sum(rewards_per_episode) / num_episodes

# Print the result
print(f"\nRandom Agent Performance over {num_episodes} episodes:")
print(f"Average Reward (Success Rate): {average_reward:.4f}")

# Close the environment
env.close()
