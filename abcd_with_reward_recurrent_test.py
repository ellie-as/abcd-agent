import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from sb3_contrib import RecurrentPPO
from abcd_with_reward_recurrent_train import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Initialise environment
    env = NGoalMazeEnv(grid_size=4, max_steps=200, sub_goal_reset_interval=200, num_rewards=4)
    model = RecurrentPPO.load("recurrent_ppo_model.zip")

    num_episodes = 100
    subgoal_counts = []
    all_hidden_states = []
    all_positions = []  # Current position (pos_id)
    all_next_goal_idx = []  # Current next goal index
    all_goal_locations = []  # Location of the current goal (row, col)
    all_rewards = []  # Rewards per step

    print("Starting testing over 100 episodes...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        lstm_states = None
        done = False
        ep_hidden_states = []
        ep_positions = []
        ep_next_goal_idx = []
        ep_goal_locations = []
        ep_rewards = []  # Track rewards for this episode
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states)
            ep_hidden_states.append(lstm_states)
            pos_id = np.argmax(obs[:-1])
            ep_positions.append(pos_id)
            next_goal_idx = env.next_goal_idx
            ep_next_goal_idx.append(next_goal_idx)
            goal_location = env.sub_goals[next_goal_idx]  # (row, col) tuple
            ep_goal_locations.append(goal_location)
            obs, reward, done, truncated, info = env.step(action)
            ep_rewards.append(reward)  # Record reward
            if done or truncated:
                break
        subgoal_count = env.goal_visits
        subgoal_counts.append(subgoal_count)
        print(f"Episode {ep + 1}: Sub-goals reached: {subgoal_count}")
        all_hidden_states.append(ep_hidden_states)
        all_positions.append(ep_positions)
        all_next_goal_idx.append(ep_next_goal_idx)
        all_goal_locations.append(ep_goal_locations)
        all_rewards.append(ep_rewards)  # Store episode rewards

    print("Testing completed over 100 episodes.")
    print("Sub-goal counts per episode:", subgoal_counts)

    # Global UMAP projection
    global_h_states = []
    episode_boundaries = [0]
    for ep_states in all_hidden_states:
        for state in ep_states:
            if state is not None:
                h = state[0][0, 0, :]  # shape: (hidden_size,)
                if isinstance(h, torch.Tensor):
                    h = h.detach().cpu().numpy()
                global_h_states.append(h)
        episode_boundaries.append(len(global_h_states))

    global_h_states = np.stack(global_h_states)
    print(f"Global hidden states shape: {global_h_states.shape}")

    # Fit UMAP on all data
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=100)
    global_embedding = reducer.fit_transform(global_h_states)
    print(f"Global UMAP embedding shape: {global_embedding.shape}")

    # Visualise global UMAP coloured by next goal index
    all_next_goal_idx_flat = np.concatenate(all_next_goal_idx)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(global_embedding[:, 0], global_embedding[:, 1],
                     c=all_next_goal_idx_flat, cmap="hsv", s=10)
    plt.colorbar(sc, label="Next Goal Index")
    plt.title("Global UMAP Colored by Next Goal Index")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

    best_indices = np.argsort(subgoal_counts)[-10:]
    print("Best episode indices:", best_indices)

    for idx in best_indices:
        start_idx = episode_boundaries[idx]
        end_idx = episode_boundaries[idx + 1]
        ep_h_states = global_h_states[start_idx:end_idx]
        pos_ids = all_positions[idx]
        ep_embedding = global_embedding[start_idx:end_idx]
        ep_rewards = all_rewards[idx]
        print(f"Episode {idx + 1} hidden states shape: {ep_h_states.shape}")

        unique_pos = sorted(set(pos_ids))
        pos_to_idx = {pos: i for i, pos in enumerate(unique_pos)}
        color_indices = [pos_to_idx[p] for p in pos_ids]
        goal_reached_indices = [t for t, r in enumerate(ep_rewards) if r == 2.0]

        plt.figure(figsize=(8, 6))
        plt.plot(ep_embedding[:, 0], ep_embedding[:, 1], marker='o', linestyle='-', color='g', alpha=0.05)
        sc = plt.scatter(ep_embedding[:, 0], ep_embedding[:, 1],
                         c=color_indices, cmap="hsv",
                         s=50, vmin=0, vmax=len(unique_pos) - 1)
        # Mark goal-reaching points
        for t in goal_reached_indices:
            plt.scatter(ep_embedding[t, 0], ep_embedding[t, 1], marker='*', color='red', s=100)
        plt.title(f"Episode {idx + 1} Hidden State Trajectory (Global UMAP)\n(Sub-goals: {subgoal_counts[idx]})")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        cbar = plt.colorbar(sc, ticks=range(len(unique_pos)))
        cbar.ax.set_yticklabels(unique_pos)
        cbar.set_label("Position ID")
        plt.show()

    all_positions_flat = np.concatenate(all_positions)
    all_next_goal_idx_flat = np.concatenate(all_next_goal_idx)
    all_goal_locations_flat = np.concatenate(all_goal_locations)  # Shape: (total_steps, 2)

    # Probe 1: Current Position
    pos_clf = LogisticRegression(max_iter=1000)
    pos_clf.fit(global_h_states, all_positions_flat)
    pos_preds = pos_clf.predict(global_h_states)
    pos_acc = accuracy_score(all_positions_flat, pos_preds)
    print("Probe accuracy for current position:", pos_acc)
    pos_weights = pos_clf.coef_  # Shape: (n_classes=16, hidden_size)
    pos_importance = np.mean(np.abs(pos_weights), axis=0)
    top_pos_dims = np.argsort(pos_importance)[-10:]
    print("Top 10 dimensions for current position:", top_pos_dims)

    # Probe 2: Current Next Goal Index
    goal_idx_clf = LogisticRegression(max_iter=1000)
    goal_idx_clf.fit(global_h_states, all_next_goal_idx_flat)
    goal_idx_preds = goal_idx_clf.predict(global_h_states)
    goal_idx_acc = accuracy_score(all_next_goal_idx_flat, goal_idx_preds)
    print("Probe accuracy for next goal index:", goal_idx_acc)
    goal_idx_weights = goal_idx_clf.coef_  # Shape: (n_classes=4, hidden_size)
    goal_idx_importance = np.mean(np.abs(goal_idx_weights), axis=0)
    top_goal_idx_dims = np.argsort(goal_idx_importance)[-10:]
    print("Top 10 dimensions for next goal index:", top_goal_idx_dims)

    # Probe 3: Location of Current Goal
    goal_loc_ids = [row * env.grid_size + col for row, col in all_goal_locations_flat]
    goal_loc_clf = LogisticRegression(max_iter=1000)
    goal_loc_clf.fit(global_h_states, goal_loc_ids)
    goal_loc_preds = goal_loc_clf.predict(global_h_states)
    goal_loc_acc = accuracy_score(goal_loc_ids, goal_loc_preds)
    print("Probe accuracy for goal location:", goal_loc_acc)
    goal_loc_weights = goal_loc_clf.coef_  # Shape: (n_classes=16, hidden_size)
    goal_loc_importance = np.mean(np.abs(goal_loc_weights), axis=0)
    top_goal_loc_dims = np.argsort(goal_loc_importance)[-10:]
    print("Top 10 dimensions for goal location:", top_goal_loc_dims)

    # Overlap Analysis
    print("Overlap between position and goal index dims:", set(top_pos_dims) & set(top_goal_idx_dims))
    print("Overlap between position and goal location dims:", set(top_pos_dims) & set(top_goal_loc_dims))
    print("Overlap between goal index and goal location dims:", set(top_goal_idx_dims) & set(top_goal_loc_dims))
