import lingam
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# including previous knowledge as a matrix
# prior knowledge matrix that have as many rows and columns as the number of columns in the dataset with values -1
def get_prior_knowledge(n: int, X):
    prior_knowledge = np.full((n, n), -1)
    # create an array length of number of columns and have as values the number of the station
    station = np.zeros(n)
    for i in range(n):
        station[i] = X.columns[i][7]

    # fill the matrix with value 0 when the station is the same
    for i in range(n):
        for j in range(n):
            if station[i] < station[j]:
                prior_knowledge[i][j] = 0
    return prior_knowledge


def restrict_unimportant_connections(n: int, low_scrap, high_scrap, prior_knowledge):
    # Perform the Mann-Whitney U Test for each pair of columns
    for i in range(n):
        _, p_value = ks_2samp(low_scrap.iloc[:, i], high_scrap.iloc[:, i])
        if p_value >= 0.05:
            for j in range(n):
                prior_knowledge[i][j] = 0

    return prior_knowledge


def fit_DirectLiNGAM(high_scrap, low_scrap, filter=True):
    X = pd.concat([high_scrap, low_scrap])
    # normalizing
    X = (X - X.min()) / (X.max() - X.min())

    n = X.shape[1]

    prior_knowledge = get_prior_knowledge(n, X)

    if filter:
        prior_knowledge = restrict_unimportant_connections(
            n, low_scrap, high_scrap, prior_knowledge
        )

    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)

    return model
