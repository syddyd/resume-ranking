import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import StructuredDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from aif360.datasets import BinaryLabelDataset

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
unprivileged_groups = [{'ethnicity': 0, 'gender': 0},{'ethnicity': 2, 'gender': 0}]  # e.g., non-white female
y_train_binary = (y_train >0.5).astype(int)
y_test_binary = (y_test>0.5).astype(int)
df_train['label'] = y_train_binary
df_test['label'] = y_test_binary
df_min = df_train[df_train['label'] == 1]
df_maj = df_train[df_train['label'] == 0]
print("Train label counts:", df_train['label'].value_counts())

print(f"Minority class count: {len(df_min)}")
print(f"Majority class count: {len(df_maj)}")


if len(df_min) == 0:
    raise ValueError("No positive (label=1) samples found in training data!")

df_min_upsampled = resample(
    df_min,
    replace=True,
    n_samples=len(df_maj),
    random_state=42

)
df_balanced = pd.concat([df_min_upsampled, df_maj])
df_train = df_balanced.sample(frac=1.0, random_state=42)

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

print(df_train.groupby(['ethnicity', 'gender'])['label'].value_counts())


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


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, group_ids_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test, group_ids_test)).batch(32)

def format_input(x, y, group_id):
    return {"tabular_features": x}, y, group_id

train_dataset = train_dataset.map(format_input)
val_dataset = val_dataset.map(format_input)

from sklearn.linear_model import LogisticRegression

X = aif_data_train.features
dummy_costs_0 = np.zeros(X.shape[0])  # simulate what might be passed

print("Unique in dummy_costs_0:", np.unique(dummy_costs_0))
print("Shape of X:", X.shape)

# Test if a LogisticRegression can even be fit
try:
    LogisticRegression().fit(X, dummy_costs_0)
except Exception as e:
    print("Error during dummy fit:", e)

print("PREPROCESSING DONE")


'''

IN PROCESSING TRAINING LOOPS FAIRNESS CLASSIFIER 
GERRYFAIR AND DIFFERENTIAL FAIRNESS


'''

class FairnessAwareRankingLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss, lambda_df=1.0, epsilon_target=0.2, num_groups=6, name="fairness_aware_loss"):
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



from scipy.special import expit  
from aif360.algorithms.inprocessing import GerryFairClassifier
from sklearn.linear_model import LogisticRegression
clf = GerryFairClassifier(predictor=LogisticRegression(solver="lbfgs", class_weight='balanced'),C=1.0, 
gamma=0.02,max_iters =50,  fairness_def='FP', printflag=False)

binary_labels = (aif_data_train.labels.flatten()>0.5).astype(int)
aif_data_train.labels=binary_labels.reshape(-1, 1)
aif_data_test.labels = (aif_data_test.labels.flatten() > 0.5).astype(int).reshape(-1, 1)


print("Train label counts:", np.unique(aif_data_train.labels, return_counts=True))
positive_count = np.sum(df_train['label'] == 1)
negative_count = np.sum(df_train['label'] == 0)
print(f"Positives: {positive_count}, Negatives: {negative_count}")

labels = aif_data_test.labels
features = aif_data_test.features
protected = aif_data_test.protected_attributes

# Build BinaryLabelDataset
binary_data_test = BinaryLabelDataset(
    favorable_label=1.0,
    unfavorable_label=0.0,
    df=pd.DataFrame(np.hstack([features, labels, protected]), 
                    columns=[f'feat_{i}' for i in range(features.shape[1])] + 
                            ['label'] + ['ethnicity', 'gender']),
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender']
)

clf.fit(aif_data_train)
gerry_binary_preds = clf.predict(binary_data_test).labels.flatten()
print("Unique prediction values:", np.unique(gerry_binary_preds))

assert np.array_equal(np.unique(gerry_binary_preds), [0, 1]), "Predictions must be binary."


preds_dataset = clf.predict(aif_data_test)
gerry_preds=preds_dataset.labels.flatten()
gerry_probs = preds_dataset.scores.flatten()# for post-processing and calibration
y_true = aif_data_test.labels.flatten()
group_ids = aif_data_test.protected_attributes.dot([10, 1])  # ethnicity * 10 + gender

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

def plot_calibration_curve(y_true, y_probs, title="Calibration Curve", filename=None):
    y_true = np.asarray(y_true).flatten()
    y_probs = np.asarray(y_probs).flatten()
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=50, pos_label=1)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

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
    # Force binary labels
    y_true = np.array(y_true).flatten()
    if y_true.max() > 1 or y_true.min() < 0:
        y_true = (y_true > 0.5).astype(int)
        
    y_probs = np.array(y_probs).flatten()

    ece = compute_ece(y_true, y_probs)
    print(f"{label} ECE: {ece:.4f}")

    if group_ids is not None:
        group_ece = subgroup_ece(y_true, y_probs, np.array(group_ids))
        for g, e in group_ece.items():
            print(f"{label} Group {g} ECE: {e:.4f}")

    plot_calibration_curve(y_true, y_probs, title=f"{label} Calibration Curve")



def subgroup_ece(y_true, y_probs, group_ids, n_bins=10):
    group_ece = {}
    unique_groups = np.unique(group_ids)
    for g in unique_groups:
        mask = group_ids == g
        if np.sum(mask) < 10:
            continue
        group_ece[g] = compute_ece(y_true[mask], y_probs[mask], n_bins)
    return group_ece



ranking_loss_fn = tfr.keras.losses.get(tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS)
ranking_metric = tfr.keras.metrics.get("ndcg",topn=5)
#Training loops
fair_loss = FairnessAwareRankingLoss(
    base_loss=base_loss_fn,
    lambda_df=1.0,
    epsilon_target=0.2,
    num_groups=6
)
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
    loss=fair_loss,
    metrics=[],
    weighted_metrics=[tfr.keras.metrics.get("ndcg",topn=5)]
)
model.fit(
    X_train,          # shape: [batch, 1, 12]
    y_train,          # shape: [batch, 1]
    sample_weight=group_ids_train,  # shape: [batch]
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test, group_ids_test)
)




#suppose preds is the name of the predictions of a model


y_logits = model.predict(X_test).flatten()
y_probs = expit(y_logits)  # scores → probabilities in [0, 1]

# Compute ECE
ece = compute_ece(y_test.flatten(), y_probs)
print(f"TF-Ranking CNN Model ECE: {ece:.4f}")

#Subgroup ECE
group_calibration = subgroup_ece(y_true, gerry_probs, group_ids)
for g, e in group_calibration.items():
    print(f"Group {g} ECE: {e:.4f}")
print("Calibration Error (ECE) - GerryFair:", ece)



evaluate_calibration(y_true, y_probs, group_ids_test, label="TF-Ranking")
evaluate_calibration(np.array(y_true), np.array(gerry_probs), np.array(group_ids), label="GerryFair")


yhat = model.predict(X_test).flatten()
accuracy = np.mean(np.abs(yhat - y_test.flatten()) < 0.002) * 100
print(f"Accuracy within 0.002 margin: {accuracy:.2f}%")



def total_loss_fn(y_true, y_pred, group_ids):
   # ranking_loss = ranking_loss_fn(y_true, y_pred)
    sample_weight = group_ids
    return fair_loss(y_true, y_pred, sample_weight)


ece = compute_ece(y_true, gerry_probs)

from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric

from aif360.algorithms.inprocessing.gerryfair.auditor import *
from PIL import Image
labels = aif_data_test.labels
features = aif_data_test.features
protected = aif_data_test.protected_attributes

# Build BinaryLabelDataset
binary_data_test = BinaryLabelDataset(
    favorable_label=1.0,
    unfavorable_label=0.0,
    df=pd.DataFrame(np.hstack([features, labels, protected]), 
                    columns=[f'feat_{i}' for i in range(features.shape[1])] + 
                            ['label'] + ['ethnicity', 'gender']),
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender']
)
binary_data_preds = binary_data_test.copy()
binary_data_preds.labels = gerry_preds.reshape(-1, 1)

metric = ClassificationMetric(
    binary_data_test,
    binary_data_preds,
    privileged_groups=[{'ethnicity': 1,'gender': 1}],
    unprivileged_groups=[{'ethnicity': 0}, {'gender': 0}, {'ethnicity': 2, 'gender': 0} ]
)

  # ethnicity * 10 + gender

# Evaluate subgroup accuracy manually
from collections import defaultdict

subgroup_acc = defaultdict(list)
for gid, y_hat, y in zip(group_ids, gerry_preds, y_true):
    subgroup_acc[gid].append(int(y_hat == y))

for gid in sorted(subgroup_acc):
    acc = np.mean(subgroup_acc[gid])
    print(f"Subgroup {gid} Accuracy: {acc:.4f}")

print("Subgroup accuracy:", metric.accuracy())
print("Disparate Impact:", metric.disparate_impact())
print("Equal opportunity difference:", metric.equal_opportunity_difference())
# Access subgroup statistics


clf.heatmapflag = True
clf.heatmap_path = 'heatmap'
clf.generate_heatmap(binary_data_test, binary_data_test.labels)
img_filename ='{}.png'.format(clf.heatmap_path)
image_ = Image.open(img_filename)
image_.show()

#blackbox auditing
gerry_metric = BinaryLabelDatasetMetric(binary_data_test)
gamma_disparity = gerry_metric.rich_subgroup(tuple(binary_data_test.labels.flatten()),'FP')
print(gamma_disparity)
#FPR VS FNR data analysis
def fp_vs_fn(dataset, gamma_list, iters):
    fp_auditor = Auditor(dataset, 'FP')
    fn_auditor = Auditor(dataset, 'FN')
    predictions_tuple = tuple(map(tuple, gerry_preds.reshape(-1, 1)))
    baseline_fp = fp_auditor.get_baseline(binary_data_test.labels.flatten(), gerry_preds)
    group_fp = fp_auditor.get_group(predictions_tuple, baseline_fp)
    fp_violations = []
    fn_violations = []
    for g in gamma_list:
        print('gamma: {} '.format(g), end =" ")
        fair_model = GerryFairClassifier(C=1.0, printflag=False, gamma=g, max_iters=iters)
        fair_model.gamma=g
        fair_model.fit(dataset)
        preds = fair_model.predict(dataset).labels.flatten()
        probs = fair_model.predict(dataset)
        fp_auditor.y_input = binary_data_test.labels.flatten()
        costs_0 = fp_auditor.compute_costs(predictions_tuple)[0]
        print("Unique costs_0 values:", np.unique(costs_0))

        plt.hist(probs.scores, bins=20)
        plt.title("Distribution of predicted probabilities")
        plt.show()

        print("Unique predicted labels:", np.unique(preds))
        print("Protected attributes shape:", dataset.protected_attributes.shape)
        print("Unique groups:", np.unique(dataset.protected_attributes, axis=0))
        print("Label distribution:", np.unique(dataset.labels, return_counts=True))



        predictions = fair_model.predict(dataset).labels.flatten().astype(int)
        _, fp_diff = fp_auditor.audit(predictions)
        _, fn_diff = fn_auditor.audit(predictions)
        fp_violations.append(fp_diff)
        fn_violations.append(fn_diff)
    print("FP violations:", fp_violations)
    print("FN violations:", fn_violations)

    plt.plot(fp_violations, fn_violations, label='adult')
    plt.xlabel('False Positive Disparity')
    plt.ylabel('False Negative Disparity')
    plt.legend()
    plt.title('FP vs FN Unfairness')
    plt.savefig('gerryfair_fp_fn.png')
    plt.close()

gamma_list = [0.0001, 0.0005, 0.001, 0.002, 0.003]
pareto_iters = 10
fp_vs_fn(binary_data_test, gamma_list, pareto_iters)
img_fn='gerryfair_fp_fn.png'
image = Image.open(img_fn)
image.show()
''''
POST PROCESSING

'''
from multicalibration import MulticalibrationPredictor


probs = expit(model.predict(X_test).flatten())     # CNN logits → probabilities 
labels = (y_test.flatten() > 0.5).astype(int)      # binary ground-truth
group_ids = group_ids_test           
              # 1D array of group IDs

# Create Boolean masks per group
unique_groups = np.unique(group_ids)
subgroups = [(group_ids == g) for g in unique_groups]  # list of bool masks
for i, mask in enumerate(subgroups):
    print(f"Group {i} mask shape: {mask.shape}, dtype: {mask.dtype}, True count: {np.sum(mask)}")
# Sanitize subgroups
sanitized_subgroups = [np.array(mask, dtype=bool).flatten() for mask in subgroups]

hkrr_params = {
    'alpha': 0.05,
    'lambda': 0.001,
    'max_iter': 200,
    'randomized': True,
    'use_oracle': False,
}

hjz_params = {
        'iterations': 200,
        'algorithm': 'OptimisticHedge',
        'other_algorithm': 'OptimisticHedge',
        'lr': 0.995,
        'other_lr': 0.995,
        'n_bins': 10,
    }

probs = np.asarray(probs).astype(np.float32).reshape(-1)
labels = np.asarray(labels).astype(np.int32).reshape(-1)

print("Shape of probs:", probs.shape)  # should be (N,)
print("Shape of labels:", labels.shape)  # should also be (N,)
print("Sample probs:", probs[:5])  # Should look like: [0.73, 0.52, ...]
print("Type of probs[0]:", type(probs[0]))  # Should be <class 'numpy.float32'>
print("Type of probs[0:1]:", type(probs[0:1]))       # ndarray
print("Type of probs[subgroups[0]][0]:", type(probs[subgroups[0]][0]))  # must also be scalar

assert probs.ndim == 1 and labels.ndim == 1
assert all(mask.shape == probs.shape for mask in sanitized_subgroups)
assert all(mask.dtype == bool for mask in sanitized_subgroups)

# Run HKRR Multicalibration
mcb = MulticalibrationPredictor('HKRR')
mcb.fit(probs, labels, sanitized_subgroups, hkrr_params)
mcb_gerry = MulticalibrationPredictor('HKRR')
mcb_gerry.fit(gerry_probs,y_true,sanitized_subgroups,hkrr_params)

# Run HJZ Multicalibration 

hjz_preds = MulticalibrationPredictor('HJZ')
hjz_preds.fit(probs, labels,sanitized_subgroups,hjz_params)
hjz_gerry = MulticalibrationPredictor('HJZ')
hjz_gerry.fit(gerry_probs, y_true, sanitized_subgroups, hjz_params)


# Get post-processed calibrated probabilities
calibrated_probs = mcb.predict(probs, subgroups)
calibrated_probs_gerry = mcb_gerry.predict(gerry_probs, sanitized_subgroups)
hjz_calibrated_probs = hjz_preds.predict(probs, sanitized_subgroups)
hjz_calibrated_gerryprobs= hjz_preds.predict(gerry_probs,sanitized_subgroups)

print("TF ECE before:", compute_ece(labels, probs))
print(" TF ECE after (HKRR):", compute_ece(labels, calibrated_probs))
print("Gerry ECE before:", compute_ece(y_true, gerry_probs))
print("Gerry ECE after (HKRR):", compute_ece(y_true, calibrated_probs_gerry))

print("TF ECE before:", compute_ece(labels, probs))
print(" TF ECE after (HJZ):", compute_ece(labels, hjz_calibrated_probs))
print("Gerry ECE before:", compute_ece(y_true, gerry_probs))
print("Gerry ECE after (HJZ):", compute_ece(y_true,hjz_calibrated_gerryprobs))