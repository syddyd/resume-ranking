import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler

# Suppose you want to protect 'race' and 'gender' at once:
# You must provide them in the features or as separate attributes
fairCV = np.load("./FairCVdb.npy", allow_pickle = True).item()
ds = fairCV['Profiles Train']
dy = fairCV['Biased Labels Train (Gender)']

ds_test = fairCV['Profiles Test']

#get rid of the facial encodings and shuffle
ds = np.delete(ds, np.s_[12:51], axis=1)
np.random.shuffle(ds)
ds = np.reshape(ds, (-1, 1, ds.shape[1]))

ds_test = np.delete(ds_test, np.s_[12:51], axis=1)
np.random.shuffle(ds_test)
ds_test = np.reshape(ds_test, (-1, 1, ds_test.shape[1]))

#get rid of data labels 
df = ds.squeeze()  # shape: (n_samples, n_features)
dt = ds_test.squeeze()

labels = np.array(dy).reshape(-1, 1)  # shape: (n_samples, 1)




# Assume ds contains features, dy are binary labels, group_ids contain race, gender, etc.

features = ds.squeeze()  # shape: (n_samples, n_features)
df = pd.DataFrame(features)
df.columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment'
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3', 'language_prof4']

dt = pd.DataFrame(dt)
dt.columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment'
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3', 'language_prof4']


df = df.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)
dt = dt.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)

scaler = StandardScaler() 
scaledData = scaler.fit_transform(X=df)
scaledTestData = scaler.transform(X=dt)
print(scaledData[:10])


#creating test and train sets for group ids 
#should return lists of groups as 3d Vectors 