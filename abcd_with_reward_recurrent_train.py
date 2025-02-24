import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def one_hot_encode(pos_idx, grid_size):
    """
    Encode an integer pos_idx into a one-hot vector of length grid_size**2.
    pos_idx = row * grid_size + col
    """
    vec = np.zeros(grid_size ** 2, dtype=np.float32)
    vec[pos_idx] = 1.0
    return vec


# N-Goal maze environment with reward included in the observation
# Like in the ABCD task, a random set of locations are chosen per trial
# Rewards are available at the locations in the specified sequence
class NGoalMazeEnv(gym.Env):
    def __init__(self, grid_size=5, max_steps=200, sub_goal_reset_interval=200, num_rewards=4):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.sub_goal_reset_interval = sub_goal_reset_interval
        self.num_rewards = num_rewards
        # The observation is the agent's state (grid_size^2) plus one extra scalar (last reward)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.grid_size ** 2 + 1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = None
        self.sub_goals = None
        self.next_goal_idx = 0
        self.current_steps = 0
        self._step_since_reset = 0
        self.visit_counts = {}
        self.goal_visits = 0
        self.last_reward = 0.0  # This will store the most recent reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._reset_sub_goals()
        self.visit_counts = {}
        self.goal_visits = 0
        all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        self.agent_pos = random.choice(all_cells)
        self.next_goal_idx = 0
        self.current_steps = 0
        self._step_since_reset = 0
        self.last_reward = 0.0  # Reset the last reward
        return self._get_observation(), {}

    def step(self, action):
        r, c = self.agent_pos
        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and r < self.grid_size - 1:
            r += 1
        elif action == 2 and c > 0:
            c -= 1
        elif action == 3 and c < self.grid_size - 1:
            c += 1
        old_pos = self.agent_pos
        self.agent_pos = (r, c)
        #print(self.agent_pos)
        self.current_steps += 1
        self._step_since_reset += 1

        # Adding this intrinsic reward seemed to help the agent to explore at the start of each episode
        # First update visit count for intrinsic bonus
        count = self.visit_counts.get(self.agent_pos, 0)
        self.visit_counts[self.agent_pos] = count + 1
        # This goes to zero after one visit to each goal
        intrinsic_bonus = 0.5 / np.sqrt(count + 1) * max(0, 1 - self.goal_visits / self.num_rewards)

        # Base reward (to incentivise shortest path)
        reward = -0.005
        if self.agent_pos == old_pos:
            reward += -0.1
        if self.agent_pos == self.sub_goals[self.next_goal_idx]:
            reward = 2.0  # Reward for reaching a goal
            self.goal_visits += 1
            print(
                f"Step {self.current_steps}: Reached goal {self.next_goal_idx} at {self.agent_pos}, Goal visits: {self.goal_visits}")
            self.next_goal_idx = (self.next_goal_idx + 1) % self.num_rewards

        reward += intrinsic_bonus
        self.last_reward = reward  # Store the reward so it can be appended to the observation

        if self._step_since_reset >= self.sub_goal_reset_interval:
            self._reset_sub_goals()
            print("Reset sub-goal locations.")
            self._step_since_reset = 0

        done = self.current_steps >= self.max_steps
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        # Current state as one-hot encoding
        row, col = self.agent_pos
        pos_idx = row * self.grid_size + col
        current_state = one_hot_encode(pos_idx, self.grid_size)
        # Concatenate the state with the last reward
        return np.concatenate([current_state, np.array([self.last_reward], dtype=np.float32)])

    def _reset_sub_goals(self):
        all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        self.sub_goals = random.sample(all_cells, self.num_rewards)


class RewardAndFinalEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_positions = []
        self.last_episode_positions = []

    def _on_training_start(self) -> None:
        self.episode_rewards.clear()
        self.current_episode_positions.clear()
        self.last_episode_positions.clear()

    def _on_step(self) -> bool:
        env_wrapped = self.training_env.envs[0]
        env_unwrapped = env_wrapped.unwrapped
        self.current_episode_positions.append(env_unwrapped.agent_pos)
        if self.locals["dones"][0]:
            self.last_episode_positions = self.current_episode_positions[:]
            self.current_episode_positions.clear()
            if self.model.ep_info_buffer:
                ep_reward = self.model.ep_info_buffer[-1]['r']
                self.episode_rewards.append(ep_reward)
        return True


class StepRewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_rewards = []
        self.step_positions = []

    def _on_training_start(self) -> None:
        self.step_rewards.clear()
        self.step_positions.clear()

    def _on_step(self) -> bool:
        r = self.locals["rewards"][0]
        self.step_rewards.append(r)
        env_wrapped = self.training_env.envs[0]
        env_unwrapped = env_wrapped.unwrapped
        self.step_positions.append(env_unwrapped.agent_pos)
        return True


def train_and_evaluate(env, timesteps=1000000):
    ep_callback = RewardAndFinalEpisodeCallback()
    step_callback = StepRewardTrackingCallback()
    callback_list = CallbackList([ep_callback, step_callback])

    policy_kwargs = {
        "net_arch": [dict(pi=[], vf=[])],
        "lstm_hidden_size": 128,
    }

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cpu',
        learning_rate=0.0003, #0.0003,
        ent_coef=0.1,
        clip_range=0.5,
        clip_range_vf=0.5
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback_list)
    training_time = time.time() - start_time

    return (model, ep_callback.episode_rewards, ep_callback.last_episode_positions,
            step_callback.step_rewards, step_callback.step_positions, training_time)


def rolling_mean(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


if __name__ == "__main__":
    env = NGoalMazeEnv(grid_size=4, max_steps=200, sub_goal_reset_interval=200, num_rewards=4)
    model, ep_rewards, final_route, step_rewards, step_positions, train_time = train_and_evaluate(env,
                                                                                                  timesteps=3000000)

    print(f"Training took {train_time:.2f} seconds.")
    print(f"Number of episodes completed: {len(ep_rewards)}")
    if ep_rewards:
        print(f"Last episode reward: {ep_rewards[-1]}")

    # Plot Episode Rewards
    plt.figure(figsize=(8, 4))
    plt.title("Episode Rewards")
    plt.plot(ep_rewards, label="Episode Reward (raw)", alpha=0.3)
    if len(ep_rewards) >= 100:
        smoothed_ep = rolling_mean(ep_rewards, 100)
        x_ep = range(100 - 1, len(ep_rewards))
        plt.plot(x_ep, smoothed_ep, label="Rolling Mean (100)", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Step-Wise Rewards
    plt.figure(figsize=(8, 4))
    plt.title("Step-Wise Rewards")
    plt.plot(step_rewards, label="Step Reward (raw)", alpha=0.3)
    window_size = 500
    if len(step_rewards) >= window_size:
        smoothed_step = rolling_mean(step_rewards, window_size)
        x_step = range(window_size - 1, len(step_rewards))
        plt.plot(x_step, smoothed_step, label=f"Rolling Mean ({window_size})", color="red", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save Model and Environment
    model.save("recurrent_ppo_model.zip")
    with open("four_goal_maze_env.pkl", "wb") as f:
        pickle.dump(env, f)
    print("Model and environment saved.")
