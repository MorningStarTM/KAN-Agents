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




def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.plot(x, running_avg, color="C1")  # Changed from scatter to plot
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    plt.show()
