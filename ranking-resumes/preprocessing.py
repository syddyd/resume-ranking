
import aif360
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np

# Assume ds contains features, dy are binary labels, group_ids contain race, gender, etc.

features = ds.squeeze()  # shape: (n_samples, n_features)
labels = np.array(dy).reshape(-1, 1)  # shape: (n_samples, 1)

# Suppose you want to protect 'race' and 'gender' at once:
# You must provide them in the features or as separate attributes

df = pd.DataFrame(features)
df['label'] = labels
df['race'] = race_column  # supply actual values
df['gender'] = gender_column  # supply actual values

# Define privileged/unprivileged groups
privileged_groups = [{'race': 1, 'gender': 1}]  # e.g., white male
unprivileged_groups = [{'race': 0, 'gender': 0}]  # e.g., non-white female

aif_data = BinaryLabelDataset(
    df=df,
    label_names=['label'],
    protected_attribute_names=['race', 'gender']
)

'''n'''