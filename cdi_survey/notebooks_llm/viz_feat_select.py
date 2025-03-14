from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
import imodelsx
from imodelsx import viz


def viz_curves(
    mets_avg: Dict[str, pd.DataFrame],
    strategies: List[str],
    outcome: str,
    n_start=0,
    n_end=1000,
    n_pecarn=8,
):
    plt.figure(dpi=150)  # , figsize=(3, 2))
    R, C = 2, 2
    COLORS = {
        "random": "gray",
        "pecarn": "tomato",
        # "gpt-3.5-turbo": "CadetBlue",
        "gpt-4-0314": "C0",
        "pecarn___gpt-4-0314": "CadetBlue",
    }

    mets = ["roc_auc", "f1", "precision", "recall"]
    for i, met in enumerate(mets):
        plt.subplot(R, C, i + 1)
        for strategy in strategies:
            assert strategy in mets_avg, f"{strategy} not in {mets_avg.keys()}"
            m = deepcopy(mets_avg[strategy])
            pecarn_val = m[met].values[n_pecarn - 1]
            m = m.loc[n_start : n_end - 1]
            if strategy not in COLORS:
                continue

            x = m["n_feats"]
            color = COLORS.get(strategy, "k")

            # mark pecarn point
            if "pecarn" in strategy:
                plt.plot(n_pecarn, pecarn_val, "o", color=color, ms=5)

            plt.plot(x, m[met], lw=2.5, label=strategy, color=color)
            plt.fill_between(
                x,
                m[met] - m[met + "_sem"],
                m[met] + m[met + "_sem"],
                alpha=0.3,
                color=color,
            )
            # plt.xlim(right=n)

        # plt.errorbar(x=np.arange(len(mets_avg)) + 1, y=mets_avg['f1'], yerr=mets_sem['f1'], lw=3, label='F1')
        plt.xlabel("# Predictors")
        plt.ylabel(imodelsx.viz.METRICS_RENAME_DICT.get(met, met))
        if i == 0:
            plt.legend()
        plt.grid(zorder=-100)
    plt.suptitle(outcome)
    plt.tight_layout()
