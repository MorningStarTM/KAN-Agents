from utils import plotLearning


class Trainer:
    def __init__(self, agent, env, epochs):
        """
        This is class for training agent.

        Args:
            agent 
            env : currently supports gym env
            epochs (int)
        """
        self.agent = agent
        self.env = env
        self.epochs = epochs
        self.history = []
        self.score = []

    
    def train(self):
        score = 0
        for i in range(self.epochs):
            done = False
            score = 0
            observation, _ = self.env.reset()

            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, _ = self.env.step(action)
                self.agent.learn(observation, reward, observation_, done)
                observation = observation
                score += reward

            self.history.append(score)
            print(f"Episode {i} Score {score}")
        
        filename = "result.png"
        plotLearning(self.history, filename=filename, window=50)