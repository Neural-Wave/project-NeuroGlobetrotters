import numpy as np
import pandas as pd
from cdt.metrics import SHD


GROUND_TRUTH = pd.read_csv("dataset/ground_truth.csv", header=None).values


def SHD_with_GT(prediction) -> np.float64:
    return SHD(GROUND_TRUTH[:15], prediction[:15])
