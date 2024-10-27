from typing import Callable

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def fit_FCI(
    X: pd.DataFrame,
    get_knowledge: Callable[[pd.DataFrame, list], BackgroundKnowledge],
    display=False,
    alpha=0.05,
) -> np.ndarray:
    g, edges = fci(X.values)

    g, edges = fci(X.values, background_knowledge=get_knowledge(X, g.get_nodes()))

    # get adjacency matrix
    adj = g.graph
    adj[adj != 0] = 1

    return adj
