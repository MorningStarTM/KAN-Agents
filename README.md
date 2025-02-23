# KAN-Agents

KAN-Agents is a deep reinforcement learning repository implementing **Kolmogorov-Arnold Networks (KAN)** with various RL agents. The project integrates KAN with reinforcement learning algorithms like **PPO and SAC** and benchmarks them across different environments.

## 📌 Implemented Algorithms
- **Proximal Policy Optimization (PPO)**
- **KAN-PPO (PPO with Kolmogorov-Arnold Network)**
- **Soft Actor-Critic (SAC) - [Work in Progress]**
- **Decision Transformer (DT) - [Work in Progress]**

## 🚀 Supported Environments
The following environments have been successfully trained using **PPO and KAN-PPO**:

- `CartPole-v1`
- `LunarLander-v3`
- `HumanoidStandup-v5`
- `Humanoid-v5`
- `InvertedDoublePendulum-v5`
- `InvertedPendulum-v5`
- `Walker2d-v5`

## 📊 PPO vs. KAN-PPO Performance Comparison
Below is the **performance comparison** of PPO and KAN-PPO across different environments. The results, including **sample efficiency**, **final performance**, and **training stability**, are stored in the `comparison` folder.

### 🔥 Performance Comparison Image:
| **CartPole** | **HumanoidStandup** |
|---------|-------------|
|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/CartPole-v1_episode_scores.png)|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/HumanoidStandup_episode_scores.png)
| **Humanoid** | **InvertedDoublePendulum** |
|---------|-------------|
|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/Humanoid_episode_scores.png)|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/InvertedDoublePendulum_episode_scores.png)
| **InvertedPendulum** | **LunarLander** |
|---------|-------------|
|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/InvertedPendulum_episode_scores.png)|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/LunarLander-v3_episode_scores.png)
| **InvertedPendulum** |
|---------|
|![App Screenshot](https://github.com/MorningStarTM/KAN-Agents/blob/eec799bbfb12b33e10642b6eb480a1a6b0bc556e/comparison/Walker2d_episode_scores.png)|


## 📂 Project Structure
```
KAN-Agents/
│── agents/                # PPO, KAN-PPO implementations
│── comparison/            # Comparison results and analysis
│── results/               # Performance metrics and logs
│── train.py               # Training scripts for agents
│── test.py                # Evaluation scripts
│── README.md              # Project documentation
```

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MorningStarTM/KAN-Agents.git
   cd KAN-Agents
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## 📌 Future Work
- **SAC with KAN** for continuous control environments
- **Integrate DreamerV3 for model-based RL**
- **Custom-designed environments** for real-world applications

## 📜 License
This project is released under the **MIT License**.

---
### 📢 Stay Updated!
⭐ **Star this repository** to stay updated on future developments!


## Acknowledgments

This project was inspired by the work of **[Blealtan](https://github.com/Blealtan)**, who implemented the **Efficient Kolmogorov–Arnold Networks (KAN) Layer**. You can check out their original repository here:

🔗 [Blealtan's GitHub Repository](https://github.com/Blealtan)