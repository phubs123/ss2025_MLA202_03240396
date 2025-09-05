import gymnasium as gym
import time

# --- EXERCISE 1: CartPole Challenge ---

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Print out the action space and observation space
print(f"Action Space: {env.action_space}")        # Discrete(2)
print(f"Observation Space: {env.observation_space}")  # Box(4,)

# === Questions and Answers ===

# Q1: What type of space is the action space? How many actions are there?
# A1: The action space is Discrete(2), meaning there are two discrete actions:
#     - 0: Push cart to the left
#     - 1: Push cart to the right

# Q2: What type of space is the observation space?
#     The output is Box(4,). This represents a continuous space with 4 float values.
#     Based on the problem, what could these four numbers possibly represent?
# A2: The 4 observations are:
#     - Cart position (how far the cart is from the center)
#     - Cart velocity (how fast the cart is moving)
#     - Pole angle (how far the pole is tilted from vertical)
#     - Pole angular velocity (how fast the pole is rotating)

# === MAIN LOOP ===

observation, info = env.reset()

terminated = False
truncated = False
total_reward = 0.0

while not terminated and not truncated:
    # Optional rendering
    env.render()

    # Randomly select an action (left or right)
    action = env.action_space.sample()
    print(f"Taking action: {action} (0:Left, 1:Right)")

    # Step the environment
    next_observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    observation = next_observation

    # Slow down rendering for visibility
    time.sleep(0.1)

# Episode completed
print(f"\nEpisode finished! Total Reward: {total_reward}")

# === Question: What does the reward represent? ===
# A3: In CartPole, you receive a reward of 1.0 for every time step the pole remains balanced.
#     So, the longer you keep the pole upright, the higher the total reward.
#     The episode ends if the pole falls or the cart goes out of bounds.

# Clean up
env.close()
