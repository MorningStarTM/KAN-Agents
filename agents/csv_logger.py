import os
import pandas as pd
import numpy as np



class CSVLogger:
    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.epochs = kwargs.get('epochs', 0)
        self.convergence_point = kwargs.get('c_point', 0)
        self.time = kwargs.get('time', 0)
        self.best_score = kwargs.get('best_score', 0.0)

    def log(self):
        if "result.csv" in os.listdir("result"):
            df = pd.read_csv("result\\result.csv")
        else:
            df = pd.DataFrame({})
            df.to_csv("result\\result.csv")
        
        data = pd.DataFrame({'agent_name': [self.agent.name],
                             'alpha':[self.agent.alpha],
                             'epochs':[self.epochs],
                             'c_point':[self.convergence_point],
                             'time':[self.time],
                             'best_score':[self.best_score]}
                             
                             )
        data = pd.concat([df, data], ignore_index=True)
        data.to_csv("result\\result.csv")