import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_learning(history, filename, window=50):
    """
    Plot the learning curve.

    Args:
        history: The list of scores.
        filename: The name of the file to save the plot.
        window: The window size for the moving average.
    """
    N = len(history)
    if N == 0:
        print("Empty history provided. No plot will be generated.")
        return
    
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(history[max(0, t - window):(t + 1)])
    
    plt.plot(running_avg)
    plt.title('Running average of previous {} scores'.format(window))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.savefig(filename)
    plt.show()


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)