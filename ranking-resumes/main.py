import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
pip install aif360
pip install cvxpy


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

'''ranking_model = tfr.keras.model.create_keras_model(
    input_creator=lambda: {"float_features": tf.keras.Input(shape=(1, 12))},
    scoring_function=model,
    loss=tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS),
    metrics=[tfr.keras.metrics.get(tfr.keras.metrics.RankingMetricKey.NDCG)],
)'''
#MAKE SURE!!!! group_ids is a flat vector!!!!!
group_ids_train = fairCV['Group IDs Train'] 
group_ids_test = fairCV['Group IDs Test']

class FairnessAwareRankingLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss, lambda_df=1.0, epsilon_target=0.0, num_groups=8, name="fairness_aware_loss"):
        super().__init__(name=name)
        self.base_loss = base_loss  
        self.lambda_df = lambda_df
        self.epsilon_target = epsilon_target
        self.num_groups = num_groups


    def differential_fairness_binary_outcome_train(self, probs):
            epsilons = []
            for i in range(tf.shape(probs)[0]):
                for j in range(tf.shape(probs)[0]):
                    if i == j:
                        continue
                    p_i = tf.clip_by_value(probs[i], 1e-6, 1 - 1e-6)
                    p_j = tf.clip_by_value(probs[j], 1e-6, 1 - 1e-6)
                    eps1 = tf.abs(tf.math.log(p_i) - tf.math.log(p_j))
                    eps2 = tf.abs(tf.math.log(1 - p_i) - tf.math.log(1 - p_j))
                    epsilons.append(tf.maximum(eps1, eps2))
            return tf.reduce_max(epsilons)

    def fairness_loss(self, prob_positive_counts, total_counts):
            concentration_param = 1.0
            alpha = concentration_param / 2
            theta = (prob_positive_counts + alpha) / (total_counts + concentration_param)
            epsilon_estimate = self.differential_fairness_binary_outcome_train(theta)
            return tf.maximum(0.0, epsilon_estimate - self.epsilon_target)
    def call(self, y_true, y_pred, sample_weight=None):
        # y_true: true relevance labels [batch_size, list_size]
        # y_pred: predicted scores [batch_size, list_size]
        # sample_weight will hold group IDs [batch_size] (custom use!)

        base = self.base_loss(y_true, y_pred)

        if sample_weight is None:
            return base  # fallback to ranking-only

        group_ids = tf.cast(sample_weight, tf.int32)
        preds_flat = tf.reshape(y_pred, [-1])
        group_ids_flat = tf.reshape(group_ids, [-1])

        # counts
        prob_positive_counts = tf.zeros((self.num_groups,), dtype=tf.float32)
        total_counts = tf.zeros((self.num_groups,), dtype=tf.float32)

        for g in range(self.num_groups):
            mask = tf.cast(tf.equal(group_ids_flat, g), tf.float32)
            total_counts = tf.tensor_scatter_nd_add(total_counts, [[g]], [tf.reduce_sum(mask)])
            pos_mask = tf.cast(preds_flat > 0.0, tf.float32)
            pos_in_group = tf.reduce_sum(pos_mask * mask)
            prob_positive_counts = tf.tensor_scatter_nd_add(prob_positive_counts, [[g]], [pos_in_group])

        fairness_penalty = self.fairness_loss(prob_positive_counts, total_counts)
        return base + self.lambda_df * fairness_penalty

# Define base loss (e.g., pairwise or listwise)
base_loss_fn = tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS)

# Fairness-aware loss
fair_loss = FairnessAwareRankingLoss(
    base_loss=base_loss_fn,
    lambda_df=1.0,
    epsilon_target=0.2,
    num_groups=8
)
model.compile(optimizer='adam', loss=fair_loss)

model.fit(
    x=train_features,        # input tensor
    y=train_labels,          # Ranking labels
    sample_weight=group_ids, # Group IDs per example
    batch_size=32,
    epochs=10,
    validation_data=(val_features, val_labels, val_group_ids)
)

from aif360.algorithms.inprocessing import GerryFairClassifier

clf = GerryFairClassifier(C=100, gamma=0.005, fairness_def='FP', printflag=False)
clf.fit(aif_data)


'''def total_loss_fn(y_true, y_pred, group_ids):
    ranking_loss = ranking_loss_fn(y_true, y_pred)
    df_penalty = differential_fairness_penalty(y_pred, group_ids)
    return ranking_loss + λ_df * df_penalty
'''
''' 
training loops 
def compute_group_counts(preds, group_ids, threshold=0.0, num_groups=8):
    """
    preds: predicted logits or probabilities [batch_size]
    group_ids: group labels [batch_size]
    """
    prob_positive_counts = tf.zeros((num_groups,), dtype=tf.float32)
    total_counts = tf.zeros((num_groups,), dtype=tf.float32)

    for g in range(num_groups):
        mask = tf.cast(tf.equal(group_ids, g), tf.float32)
        total_counts = tf.tensor_scatter_nd_add(total_counts, [[g]], [tf.reduce_sum(mask)])
        pos_mask = tf.cast(preds > threshold, tf.float32)
        pos_in_group = tf.reduce_sum(pos_mask * mask)
        prob_positive_counts = tf.tensor_scatter_nd_add(prob_positive_counts, [[g]], [pos_in_group])

    return prob_positive_counts, total_counts


    IN TRAINING 
with tf.GradientTape() as tape:
    preds = model(batch_x, training=True)
    ranking_loss = ranking_loss_fn(batch_y, preds)

    # Estimate group counts
    p_counts, t_counts = compute_group_counts(preds, batch_group_ids, num_groups=num_groups)

    df_penalty = fairness_loss(p_counts, t_counts, epsilon_target=0.2)
    loss = ranking_loss + λ_df * df_penalty

# Predict
preds = clf.predict(aif_data)

# Evaluate subgroup accuracy
from aif360.metrics import ClassificationMetric

metric = ClassificationMetric(
    aif_data,
    preds,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("Subgroup accuracy:", metric.accuracy())
print("Disparate Impact:", metric.disparate_impact())
print("Equal opportunity difference:", metric.equal_opportunity_difference())
# Access subgroup statistics
clf.classifier.subgroup_performance  # dictionary of group stats (accuracy, FP rate, etc.)

'''
