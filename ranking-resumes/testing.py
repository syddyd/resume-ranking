'''testing file to ensure that we are importing the dataset properly and can label it'''



import numpy as np 
import pandas as pd
print("You have imported things ill tell you that")
fairCV = np.load("./data/FairCVdb.npy", allow_pickle = True).item()
ds = fairCV['Profiles Train']
dy = fairCV['Biased Labels Train (Gender)']
print(ds.shape)
ds = np.delete(ds, np.s_[12:51], axis=1)
np.random.shuffle(ds)
print(ds.shape)
ds = np.reshape(ds, (-1, 1, ds.shape[1]))
print(ds.shape)
