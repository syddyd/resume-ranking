import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import StructuredDataset
from sklearn.preprocessing import StandardScaler

# Suppose you want to protect 'race' and 'gender' at once:
# You must provide them in the features or as separate attributes
fairCV = np.load("./data/FairCVdb.npy", allow_pickle = True).item()
ds = fairCV['Profiles Train']
dy = fairCV['Biased Labels Train (Gender)']

ds_test = fairCV['Profiles Test']
dy_test = fairCV['Biased Labels Train (Gender)']

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

#use sklearn standard scaler to scale the data 
scaler = StandardScaler() 
scaledData = scaler.fit_transform(X=df)
scaledTestData = scaler.transform(X=dt)

# Assume ds contains features, dy are binary labels, group_ids contain race, gender, etc.

features = ds.squeeze()  # shape: (n_samples, n_features)
labels = np.array(dy).reshape(-1, 1)  # shape: (n_samples, 1)
df = pd.DataFrame(features)
df['label'] = labels

ethnicity_train_raw = fairCV['Profiles Train'][:, 0]
print("Unique ethnicity values found:", np.unique(ethnicity_train_raw))
gender_train = fairCV['Profiles Train'][:, 1]
ethnicity_test_raw = fairCV['Profiles Test'][:, 0]
gender_test = fairCV['Profiles Test'][:, 1]

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
scaledData.columns = ['ethnicity', 'gender', 'occupation', 'suitability', 'educ_attainment',
              'prev_exp', 'reccomendation', 'availability', 'language_prof0', 'language_prof1', 'language_prof2', 
                'language_prof3', 'group_id', 'label']

# Remap ethnicity to contiguous values: 0 → 0, 1 → 1, 3 → 2
#ethnicity_map = {0: 0, 1: 1, 3: 2}
#ethnicity_train = np.vectorize(ethnicity_map.get)(ethnicity_train_raw)
#ethnicity_test = np.vectorize(ethnicity_map.get)(ethnicity_test_raw)
#group_ids_train = gender_train.astype(int) * 3 + ethnicity_train
#group_ids_test = gender_test.astype(int) * 3 + ethnicity_test

dy = np.array(dy).reshape(-1,1)
dy_test = np.array(dy_test).reshape(-1,1)
# Define privileged/unprivileged groups
privileged_groups = [{'ethnicity': 1, 'gender': 1}]  # e.g., white male
unprivileged_groups = [{'ethnicity': 0, 'gender': 0}]  # e.g., non-white female

train_features = {"tabular_features": ds}
val_features = {"tabular_features": ds_test}

aif_data = StructuredDataset(
    df=scaledData,
    label_names=['label'],
    protected_attribute_names=['group_id']
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


print("PREPROCESSING DONE")

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
# Final model
model = tf.keras.Model(inputs=inputs, outputs=x)

ranking_model = tfr.keras.model.create_keras_model(
    input_creator=lambda: {"float_features": tf.keras.Input(shape=(1, 12))},
    scoring_function=model,
    loss=tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS),
    metrics=[tfr.keras.metrics.get(tfr.keras.metrics.RankingMetricKey.NDCG)],
)

train_dataset = tf.data.Dataset.from_tensor_slices((ds, dy, group_ids_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

#MAKE SURE!!!! group_ids is a flat vector!!!!!
'''3!= 6 intersectional groups'''
num_ethnicities = 3  # G1, G2, G3 (after mapping)
group_ids = df['gender'] * num_ethnicities + ethnicity_norm
group_ids_train = fairCV['Group IDs Train'] 
group_ids_test = fairCV['Group IDs Test']

'''

IN PROCESSING TRAINING LOOPS FAIRNESS CLASSIFIER 
GERRYFAIR AND DIFFERENTIAL FAIRNESS


'''

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

from aif360.algorithms.inprocessing import GerryFairClassifier

clf = GerryFairClassifier(C=100, gamma=0.005, fairness_def='FP', printflag=False)
clf.fit(aif_data)
gerry_probs = clf.predict(aif_data)

def compute_ece(y_true, y_probs, n_bins=10):
    """Compute Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_probs > bin_boundaries[i]) & (y_probs <= bin_boundaries[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_probs[bin_mask])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece



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
'''

Calibration Post Training 


'''
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

class CalibrationPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, group_ids=None, n_bins=10):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.group_ids = group_ids
        self.n_bins = n_bins

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val).flatten()
        y_true = self.y_val.flatten()

        # Apply sigmoid if needed
        from scipy.special import expit
        y_prob = expit(y_pred)

        self.plot_calibration(y_true, y_prob, epoch)

        if self.group_ids is not None:
            self.plot_subgroup_ece(y_true, y_prob, self.group_ids)

    def plot_calibration(self, y_true, y_prob, epoch):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'Calibration Curve - Epoch {epoch+1}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Empirical Probability')
        plt.legend()
        plt.grid()
        plt.savefig(f'calibration_epoch_{epoch+1}.png')
        plt.close()

    def plot_subgroup_ece(self, y_true, y_prob, group_ids):
        from collections import defaultdict
        import numpy as np

        group_ids = np.array(group_ids)
        eces = defaultdict(float)

        for g in np.unique(group_ids):
            mask = group_ids == g
            if np.sum(mask) < 10:
                continue
            ece = compute_ece(y_true[mask], y_prob[mask])
            print(f"Epoch Subgroup {g} ECE: {ece:.4f}")


def evaluate_calibration(y_true, y_probs, group_ids=None, label="Model"):
    ece = compute_ece(y_true, y_probs)
    print(f"{label} Calibration Error (ECE): {ece:.4f}")
    
    if group_ids is not None:
        group_ece = subgroup_ece(y_true, y_probs, group_ids)
        for g, e in group_ece.items():
            print(f"{label} Group {g} ECE: {e:.4f}")
    
    plot_calibration(y_true, y_probs, label=label)


def subgroup_ece(y_true, y_probs, group_ids, n_bins=10):
    group_ece = {}
    unique_groups = np.unique(group_ids)
    for g in unique_groups:
        mask = group_ids == g
        ece_g = compute_ece(y_true[mask], y_probs[mask], n_bins)
        group_ece[g] = ece_g
    return group_ece



ranking_loss_fn = tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS)
#Training loops
fair_loss = FairnessAwareRankingLoss(
    base_loss=base_loss_fn,
    lambda_df=1.0,
    epsilon_target=0.2,
    num_groups=8
)

λ_df = 1.0
for batch_x, batch_y, batch_group_ids in train_dataset:
    with tf.GradientTape() as tape:
        preds = model(batch_x, training=True)
        ranking_loss = ranking_loss_fn(batch_y, preds)

        # Estimate group counts
        p_counts, t_counts = compute_group_counts(preds, batch_group_ids, num_groups=num_groups)

        df_penalty = fair_loss(p_counts, t_counts, epsilon_target=0.2)
        loss = ranking_loss + λ_df * df_penalty

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#suppose preds is the name of the predictions of a model

from scipy.special import expit  # Sigmoid function

y_pred_raw = model(x)  # your TF model predictions (logits or scores)
y_pred_probs = expit(y_pred_raw)  # convert scores to probabilities
ece = compute_ece(dy_test, y_pred_probs)
print("Calibration Error (ECE) - TF Ranking:", ece)

evaluate_calibration(dy_test, y_pred_probs, group_ids_test, label="TF-Ranking")
evaluate_calibration(np.array(df), np.array(gerry_probs), np.array(group_ids_test), label="GerryFair")



model.compile(optimizer='adam', loss=fair_loss)

model.fit(
    x=train_features,        # input tensor
    y=dy,          # Ranking labels
    sample_weight=group_ids, # Group IDs per example
    batch_size=32,
    epochs=10,
    validation_data=(val_features, dy_test, group_ids_test),
    callbacks=[CalibrationPlotCallback(val_inputs, dy_test, group_ids=group_ids_test)])



def total_loss_fn(y_true, y_pred, group_ids):
   # ranking_loss = ranking_loss_fn(y_true, y_pred)
    sample_weight = group_ids
    return fair_loss(y_true, y_pred, sample_weight)


ece = compute_ece(np.array(df), np.array(gerry_probs))

group_calibration = subgroup_ece(np.array(df), np.array(preds), np.array(group_ids))
for g, e in group_calibration.items():
    print(f"Group {g} ECE: {e:.4f}")
print("Calibration Error (ECE) - GerryFair:", ece)

'''
print("Subgroup accuracy:", metric.accuracy())
print("Disparate Impact:", metric.disparate_impact())
print("Equal opportunity difference:", metric.equal_opportunity_difference())
# Access subgroup statistics'''
clf.classifier.subgroup_performance  # dictionary of group stats (accuracy, FP rate, etc.)
