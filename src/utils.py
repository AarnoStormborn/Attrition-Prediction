import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from box import ConfigBox

def read_config(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    data = ConfigBox(data)
    return data    

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, cmap='Blues', cbar=False, annot=True, fmt='.3g', ax=ax)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    return fig
