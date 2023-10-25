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

def plot_confusion_matrix(filepath, confusion_matrix):
    plt.figure()
    sns.heatmap(confusion_matrix, cmap='Blues', cbar=False, annot=True, fmt='.3g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(filepath, 'confusion_matrix.png'))
