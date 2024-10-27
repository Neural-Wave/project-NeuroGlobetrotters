import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from sklearn.preprocessing import MinMaxScaler
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# Load and preprocess the dataset
data = pd.read_csv("dataset/low_scrap.csv")

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled, columns=data.columns)
n = data.shape[1]

g, edges = fci(data.values)

# g, edges = fci(data.values, background_knowledge=get)
