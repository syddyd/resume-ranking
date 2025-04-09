import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 

fairCV = np.load("FairCVdb.npy", allow_pickle = True).item()
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