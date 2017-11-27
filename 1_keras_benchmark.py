import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

train = pd.read_table("./data/train.tsv")
test = pd.read_table("./data/test.tsv")
print(train.shape)
print(test.shape)
train.head(2)
test.head(2)
