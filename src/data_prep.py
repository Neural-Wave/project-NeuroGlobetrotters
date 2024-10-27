import numpy as np
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import pandas as pd


def load_prediction_from_csv(file_path: str):
    # Adjust header based on your CSV structure
    data = pd.read_csv(file_path, header=0, index_col=0)
    return data.values  # Convert to numpy array


def no_backedge_knowledge(X: pd.DataFrame, nodes) -> BackgroundKnowledge:
    n = X.shape[1]
    bk = BackgroundKnowledge()

    # knowledge encoding
    for i in range(n):
        # no self loops
        bk.add_forbidden_by_node(nodes[i], nodes[i])
        for j in range(n):
            # don't allow the station to influence the previous
            if int(X.columns[i][7]) > int(X.columns[j][7]):
                bk.add_forbidden_by_node(nodes[i], nodes[j])

    return bk


def get_knowledge_from_csv(filename: str, **kwargs):
    df = pd.read_csv(filename, **kwargs)

    def knowledge_from_csv(X: pd.DataFrame, nodes) -> BackgroundKnowledge:
        bk = BackgroundKnowledge()
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if value == 0:
                    bk.add_forbidden_by_node(nodes[i], nodes[j])
                if value == 1:
                    bk.add_required_by_node(nodes[i], nodes[j])
        return bk

    return knowledge_from_csv


# def prior_knowledge_matrix(columns):
#     """
#     prior knowledge matrix for LiNGAM where:
#     0: no directed path possible (temporal constraint violation)
#     1: directed path
#     -1: no prior knowledge (we'll allow the algorithm to determine)
#     """
#     n_features = len(columns)
#     prior_knowledge = np.full((n_features, n_features), -1)
#
#     # get station number
#     def get_station_number(col_name):
#         return int(col_name.split("_")[0].replace("Station", ""))
#
#     # get measurement point number
#     def get_mp_number(col_name):
#         return int(col_name.split("_")[2])
#
#     for i in range(n_features):
#         for j in range(n_features):
#             station_i = get_station_number(columns[i])
#             station_j = get_station_number(columns[j])
#
#             # constraint
#             if station_i > station_j:
#                 prior_knowledge[i, j] = 0
#
#             # should we allow internal dependencies?
#             # if station_i == station_j:
#             #     prior_knowledge[i, j] = -1  # Let LiNGAM determine
#
#     # No self loop allowed
#     np.fill_diagonal(prior_knowledge, 0)
#
#     return prior_knowledge
