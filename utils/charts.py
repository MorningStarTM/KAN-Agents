import matplotlib.pyplot as plt
import numpy as np


def generate_comparison_charts(dqn_results, kqn_results, filename_prefix):
    """
    Generate comparison charts for DQN and KQN.
    
    :param dqn_results: Tuple of (scores, epsilons) for DQN
    :param kqn_results: Tuple of (scores, epsilons) for KQN
    :param filename_prefix: Prefix for the chart filenames
    """
    # Unpack results
    dqn_scores, dqn_epsilons = dqn_results
    kqn_scores, kqn_epsilons = kqn_results

    # Generate x-axis values
    episodes = [i + 1 for i in range(len(dqn_scores))]

    # 1. Total Reward over Episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, dqn_scores, label="DQN", alpha=0.7)
    plt.plot(episodes, kqn_scores, label="KQN", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward over Episodes")
    plt.legend()
    plt.savefig(f"{filename_prefix}_total_reward.png")
    plt.show()

    # 2. Sample Efficiency (Cumulative Reward per Episode)
    dqn_cumulative_reward = np.cumsum(dqn_scores) / episodes
    kqn_cumulative_reward = np.cumsum(kqn_scores) / episodes

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, dqn_cumulative_reward, label="DQN", alpha=0.7)
    plt.plot(episodes, kqn_cumulative_reward, label="KQN", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward per Episode")
    plt.title("Sample Efficiency")
    plt.legend()
    plt.savefig(f"{filename_prefix}_sample_efficiency.png")
    plt.show()

    # 3. Final Performance (Average Reward over Last N Episodes)
    N = 100  # Define the number of episodes to calculate final performance
    dqn_final_performance = np.mean(dqn_scores[-N:])
    kqn_final_performance = np.mean(kqn_scores[-N:])

    plt.figure(figsize=(10, 6))
    plt.bar(["DQN", "KQN"], [dqn_final_performance, kqn_final_performance], alpha=0.7)
    plt.ylabel("Final Performance (Average Reward)")
    plt.title(f"Final Performance (Last {N} Episodes)")
    plt.savefig(f"{filename_prefix}_final_performance.png")
    plt.show()

    # 4. Training Stability (Variance in Scores)
    dqn_variance = np.var(dqn_scores[-N:])
    kqn_variance = np.var(kqn_scores[-N:])

    plt.figure(figsize=(10, 6))
    plt.bar(["DQN", "KQN"], [dqn_variance, kqn_variance], alpha=0.7)
    plt.ylabel("Training Stability (Variance in Reward)")
    plt.title(f"Training Stability (Last {N} Episodes)")
    plt.savefig(f"{filename_prefix}_training_stability.png")
    plt.show()
