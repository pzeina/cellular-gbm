import torch
from torch import nn
from torch.distributions import Categorical
import gymnasium as gym
import tissue_env as tissue_env
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
env = gym.make("tissue_env/TerrainWorld-v0") #, render_mode="human")

action_size = env.action_space.n
state, info = env.reset()

# Convert the state to a tensor
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
state_size = state_tensor.shape[1]

# Set up policy network and optimizer
print(f"State size: {state_size}, Action size: {action_size}")
policy_net = PolicyNetwork(state_size, action_size).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-3)

NUMBER_OF_EPISODES = 1000
GAMMA = 0.99
EPSILON = 0.1  # Epsilon for epsilon-greedy strategy

total_rewards = []

for episode in range(NUMBER_OF_EPISODES):
    state, info = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    episode_reward = 0

    while True:
        # Get action probabilities from policy network
        action_probs = policy_net(state_tensor)
        action_dist = Categorical(logits=action_probs)

        # Epsilon-greedy action selection
        if torch.rand(1).item() < EPSILON:
            action = torch.tensor(env.action_space.sample())
        else:
            action = action_dist.sample()

        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action.item())
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

        # Calculate loss and perform backpropagation
        log_prob = action_dist.log_prob(action)
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state_tensor = next_state_tensor
        episode_reward += reward

        if terminated or truncated:
            total_rewards.append(episode_reward)
            break

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")

print("Training finished.")

plt.plot(total_rewards)
plt.show()