import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import StructuredDataset
from sklearn.preprocessing import StandardScaler
from metrics import *


fairCV = np.load("./data/FairCVdb.npy", allow_pickle=True).item()
X_train_raw = np.delete(fairCV['Profiles Train'], np.s_[12:51], axis=1)  # Remove embeddings
X_test_raw = np.delete(fairCV['Profiles Test'], np.s_[12:51], axis=1)

y_train = np.array(fairCV['Biased Labels Train (Gender)']).reshape(-1, 1)
y_test = np.array(fairCV['Blind Labels Test']).reshape(-1, 1)

rng = np.random.default_rng(seed=42)
rng.shuffle(X_train_raw)
rng.shuffle(X_test_raw)

# === Reshape for CNN Input [batch_size, 1, 12] ===
X_train = X_train_raw.reshape(-1, 1, 12).astype(np.float32)
X_test = X_test_raw.reshape(-1, 1, 12).astype(np.float32)

ethnicity_train = fairCV['Profiles Train'][:, 0]
print("Unique ethnicity values found:", np.unique(ethnicity_train))
gender_train = fairCV['Profiles Train'][:, 1]
ethnicity_test = fairCV['Profiles Test'][:, 0]
gender_test = fairCV['Profiles Test'][:, 1]


group_ids_train = (ethnicity_train * 10 + gender_train).astype(int)
group_ids_test = (ethnicity_test * 10 + gender_test).astype(int)
#print("Shape of group ids_test:" , group_ids_test.shape)
columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
           'prev_exp', 'recommendation', 'availability',
           'language_prof0', 'language_prof1', 'language_prof2', 
           'language_prof3']

# === Create pandas DataFrames for StructuredDataset ===
df_train = pd.DataFrame(X_train_raw, columns=columns)
df_test = pd.DataFrame(X_test_raw, columns=columns)

'''
#use sklearn standard scaler to scale the data 
scaler = StandardScaler() 
scaledData = scaler.fit_transform(X=df)
scaledTestData = scaler.transform(X=dt)
'''
# Assume ds contains features, dy are binary labels, group_ids contain race, gender, etc.


df_train['label'] = y_train
df_test['label'] = y_test
df_train['group_id'] = group_ids_train
df_test['group_id'] = group_ids_test

'''
df = df.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)
dt = dt.assign(group_id = lambda x : x.ethnicity * 10 + x.gender)
'''
# Define privileged/unprivileged groups
feature_cols = [col for col in columns if col not in ['ethnicity', 'gender']]
scaler = StandardScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_test[feature_cols] = scaler.transform(df_test[feature_cols])
privileged_groups = [{'ethnicity': 1, 'gender': 1}]  # e.g., white male
unprivileged_groups = [{'ethnicity': 0, 'gender': 0}]  # e.g., non-white female


aif_data_train = StructuredDataset(
    df=df_train,
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender']
)

aif_data_test = StructuredDataset(
    df=df_test,
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender']
)

for i in range(5):
    print(f"{X_train[i]} label: {y_train[i]}")


''' fairness models'''
# Inputs: shape [batch_size, 1, 12] after reshaping
inputs = tf.keras.Input(shape=(1, 12), name="tabular_features")

# --- CNN Block ---
# Simulate convolution over the "feature channels"
x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(inputs)
x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(1)(x)
'''Optional 
x = tf.keras.layers.Conv1D(32, 1, activation='relu')(inputs)
x = tf.keras.layers.Conv1D(64, 1, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x) -> if we want regularization 
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(1)(x)

'''
# Define base loss (e.g., pairwise or listwise)
base_loss_fn = tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS)

# Final model
model = tf.keras.Model(inputs=inputs, outputs=x)
#MAKE SURE!!!! group_ids is a flat vector!!!!!
'''3!= 6 intersectional groups
num_ethnicities = 3  # G1, G2, G3 (after mapping)
group_ids_train = fairCV['Group IDs Train'] 
group_ids_test = fairCV['Group IDs Test']
'''
'''ranking_model = tfr.keras.model.create_keras_model(
    input_creator=lambda: {"float_features": tf.keras.Input(shape=(1, 12))},
    scoring_function=model,
    loss=tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS),
    metrics=[tfr.keras.metrics.get(tfr.keras.metrics.RankingMetricKey.NDCG)],
)'''

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, group_ids_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, group_ids_test)).batch(32)

def format_input(x, y, group_id):
    return {"tabular_features": x}, y, group_id

train_dataset = train_dataset.map(format_input)
val_dataset = val_dataset.map(format_input)




print("PREPROCESSING DONE")
