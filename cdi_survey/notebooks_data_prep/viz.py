import os.path
from os.path import join as oj

import matplotlib.pyplot as plt

path_to_current_file = os.path.dirname(os.path.abspath(__file__))


def savefig(fname):
    plt.tight_layout()
    plt.savefig(oj(path_to_current_file, '..', 'results', fname + '.png'), dpi=300, bbox_inches="tight")
    plt.savefig(oj(path_to_current_file, '..', 'results', fname + '.pdf'), bbox_inches="tight")
