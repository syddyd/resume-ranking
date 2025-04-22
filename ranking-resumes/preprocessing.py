import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler

# Suppose you want to protect 'race' and 'gender' at once:
# You must provide them in the features or as separate attributes
fairCV = np.load("./data/FairCVdb.npy", allow_pickle = True).item()
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
df.columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3']

dt = pd.DataFrame(dt)
dt.columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3']


df = df.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)
df = df.assign(labels = fairCV['Blind Labels Train'])
dt = dt.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)
dt = dt.assign(labels = fairCV['Blind Labels Test'])

scaler = StandardScaler() 
scaledData = scaler.fit_transform(X=df)
scaledTestData = scaler.transform(X=dt)

scaledData = pd.DataFrame(scaledData)
lbls = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3', 'group_id', 'label']
scaledData.columns = lbls

scaledTestData = pd.DataFrame(scaledTestData)
scaledTestData = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3', 'group_id', 'label']

df_final = np.array()
for label in lbls : 
    df_final = np.append(df_final,  )

gender_col = scaledData[:,1]

print(gender_col)


#creating test and train sets for group ids 
#should return lists of groups as 3d Vectors 





from aif360.datasets import StructuredDataset

aif_data = StructuredDataset(
    df=df,
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender'],
    privileged_classes=[{'ethnicity': 1}, {'gender': 1}],
    features_to_drop=[]  
)
