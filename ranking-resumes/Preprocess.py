
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

# Suppose you want to protect 'race' and 'gender' at once:
# You must provide them in the features or as separate attributes
fairCV = np.load("./data/FairCVdb.npy", allow_pickle = True).item()
ds = fairCV['Profiles Train']
dy = fairCV['Biased Labels Train (Gender)']
print(ds.shape)
ds = np.delete(ds, np.s_[12:51], axis=1)
np.random.shuffle(ds)
print(ds.shape)
ds = np.reshape(ds, (-1, 1, ds.shape[1]))
#print(ds.shape)

ds_test = fairCV['Profiles Test']
ds_test = np.delete(ds_test, np.s_[12:51], axis=1)
ds_test = np.reshape(ds_test, (-1, 1, ds_test.shape[1]))
dy_test = fairCV['Blind Labels Test']

# Assume ds contains features, dy are binary labels, group_ids contain race, gender, etc.

features = ds.squeeze()  # shape: (n_samples, n_features)
labels = np.array(dy).reshape(-1, 1)  # shape: (n_samples, 1)
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


for i in range(0,5):
  print(f"{ds[i]} label: {dy[i]}")

# Create a model
inputs = {
    "float_features": tf.keras.Input(shape=(None, 12), dtype='float32')
}
norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
x = tf.concat(norm_inputs, axis=-1)
for layer_width in [128, 64, 32]:
  x = tf.keras.layers.Dense(units=layer_width)(x)
  x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

# Compile and train
model = tf.keras.Model(inputs=inputs, outputs=scores)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tfr.keras.losses.MeanSquaredLoss(),
    metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"))
model.fit(ds, dy, epochs=3)

yhat = model.predict(ds_test)
acc = 0 
all = 0
for i in range(len(yhat)): 
  if abs(yhat[i] - dy_test[i]) < 0.002 : 
    acc +=1
  all +=1

print(f"accuracy: {100*(acc/all)}%")

''' fairness models'''
# Inputs: shape [batch_size, 1, 12] after reshaping
inputs = tf.keras.Input(shape=(1, 12), name="float_features")

# --- CNN Block ---
# Simulate convolution over the "feature channels"
x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(inputs)
x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)

# --- DNN Head ---
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)  # Output ranking score


# Final model
model = tf.keras.Model(inputs=inputs, outputs=x)

ranking_model = tfr.keras.model.create_keras_model(
    input_creator=lambda: {"float_features": tf.keras.Input(shape=(1, 12))},
    scoring_function=model,
    loss=tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS),
    metrics=[tfr.keras.metrics.get(tfr.keras.metrics.RankingMetricKey.NDCG)],
)

#MAKE SURE!!!! group_ids is a flat vector!!!!!
group_ids_train = fairCV['Group IDs Train'] 
group_ids_test = fairCV['Group IDs Test']



'''n'''