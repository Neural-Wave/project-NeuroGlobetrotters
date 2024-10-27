from typing import Callable
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def fit_PC(
    X: pd.DataFrame,
    get_knowledge: Callable[[pd.DataFrame, list], BackgroundKnowledge],
    display=False,
    alpha=0.05,
) -> np.ndarray:
    cg = pc(X.values, alpha=alpha)
    nodes = cg.G.get_nodes()

    cg = pc(X.values, background_knowledge=get_knowledge(X, nodes), alpha=alpha)

    # visualization using pydot
    if display:
        cg.draw_pydot_graph()

    # get adjacency matrix
    adj = cg.G.graph
    adj[adj != 0] = 1

    return adj
