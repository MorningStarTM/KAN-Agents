import os
import pandas as pd
import numpy as np



class CSVLogger:
    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.epochs = kwargs.get('epochs', 0)
        self.convergence_point = kwargs.get('c_point', 0)

    def log(self):
        data = pd.DataFrame({'f1':self.agent.fc1_dims,
                             'f2':self.agent.fc2_dims,
                             'alpha':self.agent.alpha,
                             'epochs':self.epochs,
                             'c_point':self.convergence_point}
                             )
        
        data.to_csv("result.csv")
