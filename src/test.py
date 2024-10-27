import sys

import numpy as np
import pandas as pd
from FCI import fit_FCI
from data_prep import get_knowledge_from_csv, no_backedge_knowledge
from evaluation import GROUND_TRUTH, SHD_with_GT
from PC import fit_PC
from cdt.metrics import SHD
from LiNGAM import fit_DirectLiNGAM

df = pd.read_csv(
    "./dataset/prior_knowledge.csv",
)

# print("SHD:", SHD(GROUND_TRUTH[:15], df.values[:15]))

# df[df == -1] = 0
#
# res = pd.read_csv(
#     "./dataset/adj_matrix_jb_LOW.csv",
#     index_col=0,
# )
# res[res != 0] = 1
#
# print(SHD_with_GT(res.values.T))

np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

np.random.seed(100)


# loading the dataset
high_scrap = pd.read_csv("dataset/high_scrap.csv")
low_scrap = pd.read_csv("dataset/low_scrap.csv")

# X = pd.concat([high_scrap, low_scrap])
X = low_scrap
# normalizing
X = (X - X.min()) / (X.max() - X.min())

# adj = fit_FCI(X, get_knowledge_from_csv("dataset/prior_knowledge.csv"))
# adj = fit_PC(X, no_backedge_knowledge)
adj = fit_PC(X, get_knowledge_from_csv("dataset/prior_knowledge.csv"))

# Score
score = SHD_with_GT(adj)
print(score)


def main():
    # np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
    np.random.seed(100)

    # loading the dataset
    high_scrap = pd.read_csv("dataset/high_scrap.csv")
    low_scrap = pd.read_csv("dataset/low_scrap.csv")

    model = fit_DirectLiNGAM(high_scrap, low_scrap)

    distance = SHD_with_GT(model.adjacency_matrix_)
    print(distance)
