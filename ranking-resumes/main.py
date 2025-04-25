import os
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np 
import pandas as pd
from aif360.datasets import StructuredDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from aif360.datasets import BinaryLabelDataset
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

fairCV = np.load("./data/FairCVdb.npy", allow_pickle=True).item()
X_train_raw = np.delete(fairCV['Profiles Train'], np.s_[12:51], axis=1)  # Remove embeddings
X_test_raw = np.delete(fairCV['Profiles Test'], np.s_[12:51], axis=1)

y_train = np.array(fairCV['Biased Labels Train (Gender)']).reshape(-1,1)
y_train_remix = (y_train>0.37).astype(int)
y_test = np.array(fairCV['Blind Labels Test']).reshape(-1, 1)

rng = np.random.default_rng(seed=42)

perm_train = rng.permutation(X_train_raw.shape[0])
X_train_raw = X_train_raw[perm_train]
y_train = y_train[perm_train]
perm_test = rng.permutation(X_test_raw.shape[0])
X_test_raw = X_test_raw[perm_test]
y_test = y_test[perm_test]

# === Reshape for CNN Input [batch_size, 1, 12] ===
X_train = X_train_raw.reshape(-1, 1, 12).astype(np.float32)
X_test = X_test_raw.reshape(-1, 1, 12).astype(np.float32)

ethnicity_train = fairCV['Profiles Train'][:, 0]
print("Unique ethnicity values found:", np.unique(ethnicity_train))
gender_train = fairCV['Profiles Train'][:, 1]
ethnicity_test = fairCV['Profiles Test'][:, 0]
gender_test = fairCV['Profiles Test'][:, 1]
ethnicity_test = ethnicity_test[perm_test]
gender_test = gender_test[perm_test]
ethnicity_train = ethnicity_train[perm_train]
gender_train = gender_train[perm_train]

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
feature_cols = df_train.columns.difference(['label', 'ethnicity', 'gender'])
scaler = StandardScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_test[feature_cols] = scaler.transform(df_test[feature_cols])
print("Feature means:", df_train[feature_cols].mean())
print("Feature stds:", df_train[feature_cols].std())
privileged_groups = [{'ethnicity': 1, 'gender': 1}]  # e.g., white male
unprivileged_groups = [{'ethnicity': 0, 'gender': 0},{'ethnicity': 2, 'gender': 0}]  # e.g., non-white female
y_train_binary = (y_train >0.37).astype(int).reshape(-1,1)
y_test_binary = (y_test>0.37).astype(int).reshape(-1,1)
df_train['label'] = y_train_binary
df_test['label'] = y_test_binary
df_min = df_train[df_train['label'] == 1]
df_maj = df_train[df_train['label'] == 0]
print("Train label counts:", df_train['label'].value_counts())

print(f"Minority class count: {len(df_min)}")
print(f"Majority class count: {len(df_maj)}")

#test delete 
df_train['group_id'] = df_train['ethnicity'] * 10 + df_train['gender']

# Display subgroup label distribution (normalized counts of 0 and 1)
subgroup_distribution = df_train.groupby('group_id')['label'].value_counts(normalize=True).unstack().fillna(0)
print("\nSubgroup label distribution (normalized):")
print(subgroup_distribution)

# Optional: show raw counts
raw_counts = df_train.groupby('group_id')['label'].value_counts().unstack().fillna(0)
print("\nSubgroup label raw counts:")
print(raw_counts)
#end of test delete

if len(df_min) == 0:
    raise ValueError("No positive (label=1) samples found in training data!")

# For each unique group_id (combination of ethnicity & gender), resample
df_group_balanced = []
for group_id in df_train['group_id'].unique():
    group_df = df_train[df_train['group_id'] == group_id]
    threshold = 0.37  # or keep configurable
    df0 = group_df[group_df['label'] <= threshold]
    df1 = group_df[group_df['label'] > threshold]
    if len(df1) == 0: continue  # Skip if no positives
    upsampled = resample(df1, replace=True, n_samples=len(df0), random_state=42)
    df_group_balanced.append(pd.concat([df0, upsampled]))

df_balanced = pd.concat(df_group_balanced).sample(frac=1.0, random_state=42)

aif_data_train = StructuredDataset(
    df=df_balanced,
    label_names=['label'],
    protected_attribute_names=['ethnicity', 'gender']
)
print("Train label counts(before training):", np.unique(aif_data_train.labels, return_counts=True))


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



X = aif_data_train.features
dummy_costs_0 = np.zeros(X.shape[0])  # simulate what might be passed

print("Unique in dummy_costs_0:", np.unique(dummy_costs_0))
print("Shape of X:", X.shape)


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


#Train logistic regression manually

X = aif_data_train.features
y = aif_data_train.labels.flatten()

#print("Training accuracy:", clf_test.score(X, y))

print("Protected attributes shape:", aif_data_train.protected_attributes.shape)
print("Unique groups:", np.unique(aif_data_train.protected_attributes, axis=0))

#BASE MODEL IS FINE THIS IS FINE DO NOT MESSS WITH CLF_TEST  PLEASE


print("Train label counts(justbefore):", np.unique(aif_data_train.labels, return_counts=True))
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
#also deleter

class ProbWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]



'''
A_p = aif_data_train.protected_attributes'''
clf_test = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
from sklearn.calibration import CalibratedClassifierCV
cal_clf = CalibratedClassifierCV(clf_test, method='sigmoid', cv=3)

X_p = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
Y_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
y_train_icky = (y_train.flatten() > 0.37).astype(int)
y_test_bin = (y_test.flatten()>0.37).astype(int)

cal_clf.fit(X_p, y_train_icky)
probs = cal_clf.predict_proba(Y_test)[:, 1]
#initial_preds = (probs > 0.3).astype(int)
initial_preds = (probs > 0.35).astype(int)
wrapped_cal_clf = ProbWrapper(cal_clf)
from sklearn.metrics import confusion_matrix

#Intersectional Group Fairness
print("Intersectional Group Fairness Metric")
df_test_analysis = df_test.copy()
df_test_analysis['pred'] = initial_preds
intersectional_groups = df_test_analysis.groupby(['ethnicity', 'gender'])

for (eth, gen), group_df in intersectional_groups:
    cm = confusion_matrix(group_df['label'], group_df['pred'])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn/ (tp + fn + 1e-6)
    print(f"Group ({eth}, {gen}): FP Rate = {fpr:.4f}")
    print(f"Group ({eth}, {gen}): FN Rate = {fnr:.4f}")

import matplotlib.pyplot as plt

# Group names (ethnicity, gender)
groups = ['(0.0, 0.0)', '(0.0, 1.0)', '(1.0, 0.0)', 
          '(1.0, 1.0)', '(2.0, 0.0)', '(2.0, 1.0)']
fp_rates = [0.1551, 0.0074, 0.1688, 0.0141, 0.1358, 0.0095]

plt.figure(figsize=(10, 6))
bars = plt.bar(groups, fp_rates, color='salmon')
plt.title('False Positive Rates by Intersectional Group (Ethnicity × Gender)', fontsize=14)
plt.ylabel('False Positive Rate')
plt.xlabel('Group (Ethnicity, Gender)')
plt.ylim(0, max(fp_rates) + 0.05)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

plt.axhline(y=sum(fp_rates)/len(fp_rates), color='gray', linestyle='--', label='Mean FP Rate')
plt.legend()
plt.tight_layout()
plt.show()


#End of that metric



for g in np.unique(group_ids_test):
    idx = group_ids_test == g
    cm = confusion_matrix(y_test_bin[idx], initial_preds[idx])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], cm[0][1], 0, 0)
    print(f"Group {g}: FP rate = {fp / (fp + tn + 1e-6):.4f}")

for g in np.unique(group_ids_test):
    plt.hist(probs[group_ids_test == g], bins=50, alpha=0.5, label=f"Group {g}")
plt.legend()
plt.title("Prediction distribution by group")
plt.show()
plt.hist(initial_preds, bins=50)
plt.title("Prediction probability distribution")
plt.show()

#Calibration Plots per Group
print("Calibration Curve by Subgroup")
for g in np.unique(group_ids_test):
    idx = group_ids_test == g
    prob_true, prob_pred = calibration_curve(y_test_bin[idx], probs[idx], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Group {g}')
    
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Predicted probability")
plt.ylabel("Observed probability")
plt.title("Calibration Curves by Group")
plt.legend()
plt.show()

#end of calibration plots

#Predictive quality across subgroups 
from sklearn.metrics import roc_auc_score
print("AUC-ROC per Group")
for g in np.unique(group_ids_test):
    idx = group_ids_test == g
    auc = roc_auc_score(y_test_bin[idx], probs[idx])
    print(f"Group {g}: AUC-ROC = {auc:.3f}")

import matplotlib.pyplot as plt

group_names = ['0', '1', '10', '11', '20', '21']
auc_scores = [0.983, 0.970, 0.983, 0.964, 0.982, 0.979]

plt.bar(group_names, auc_scores, color='skyblue')
plt.ylim(0.95, 1.0)
plt.ylabel("AUC-ROC")
plt.xlabel("Subgroup")
plt.title("AUC-ROC by Subgroup")
plt.axhline(y=0.5, color='red', linestyle='--', label="Random Guessing")
plt.legend()
plt.show()
# end of that 

clf = GerryFairClassifier(predictor=wrapped_cal_clf,C=1.0, 
gamma=0.06, max_iters =100,  fairness_def='FP', printflag=True)

'''
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
baseline_clf = LogisticRegression(solver='lbfgs', class_weight='balanced')
baseline_clf.fit(X_p, y)
initial_preds = baseline_clf.predict(X_p)
auditor = Auditor(aif_data_train,'FP')  # Correct
metric_baseline = auditor.get_baseline(y, initial_preds)

# Get group violating fairness the most
group = auditor.get_group(initial_preds, metric_baseline)
preds_tuple = tuple(map(tuple, np.zeros_like(aif_data_train.labels)))  # or try ones
fairness_violation, _ = auditor.audit(preds_tuple)
'''

#deletaroni
print("Final label distribution:", np.unique(aif_data_train.labels, return_counts=True))
print("Prediction distribution before fairness model:", np.unique(initial_preds, return_counts=True))

print("Binary_data_test label values:", np.unique(binary_data_test.labels))
clf.fit(aif_data_train)

#test
predictions = clf.predict(aif_data_train).labels.flatten()
print("Unique predictions aif_data_train:", np.unique(predictions), "should not be 0")
preds = clf.predict(aif_data_train).labels.flatten()
print("Train predictions aif_data_train:", np.unique(preds, return_counts=True))
#test
test_preds = clf.predict(binary_data_test).labels.flatten()
print("Test predictions binary_data_test:", np.unique(test_preds, return_counts=True))


gerry_binary_preds_dataset = clf.predict(binary_data_test)
gerry_probs_ = gerry_binary_preds_dataset.scores.flatten()
print("Min/max predicted scores:", gerry_probs_.min(), gerry_probs_.max())
print("Distribution summary:", np.percentile(gerry_probs_, [0, 25, 50, 75, 100]))
threshold = 0.37
gerry_binary_preds = (gerry_probs_ > threshold).astype(int).reshape(-1, 1)  # Make sure it's column-shaped!
binary_data_preds = binary_data_test.copy()
binary_data_preds.labels = gerry_binary_preds


#assert set(np.unique(binary_data_preds.labels)) <= {0, 1}, "Predictions are not binary!"

print("Unique prediction values after binary_predict with gerry_binary preds:", np.unique(binary_data_preds.labels))


from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor

auditor = Auditor(dataset=binary_data_test, fairness_def='FP')
predictions_tuple = tuple((int(p),) for p in gerry_binary_preds.flatten())
group, fp_diff = auditor.audit(predictions_tuple)

print("Auditor FP violation:", fp_diff)
print("Violated group:", group)

#Fairness accuracy tradeoff

gammas = [0.03, 0.04, 0.05, 0.06, 0.07]
accs = []
violations = []

for gamma in gammas:
    clf = GerryFairClassifier(predictor=wrapped_cal_clf, gamma=gamma, fairness_def='FP', max_iters=100)
    clf.fit(aif_data_train)
    preds = clf.predict(binary_data_test).labels.flatten()
    accs.append(np.mean(preds == y_test_bin))
    # run auditor here
    predictions_tuple = tuple((int(p),) for p in preds)
    _, fp_diff = auditor.audit(predictions_tuple)
    violations.append(fp_diff)

plt.plot(gammas, accs, label="Accuracy", marker='o')
plt.plot(gammas, violations, label="FP Violation", marker='x')
plt.xlabel("Gamma")
plt.ylabel("Metric")
plt.title("Fairness vs. Accuracy Tradeoff")
plt.legend()
plt.xscale("log")
plt.show()

#



def compute_fp_rates(dataset, predictions):
    df = dataset.convert_to_dataframe()[0]
    df['pred'] = predictions
    protected_attrs = df[['ethnicity', 'gender']]
    groups = protected_attrs.drop_duplicates().values

    fp_rates = {}

    for group in groups:
        mask = (df['ethnicity'] == group[0]) & (df['gender'] == group[1])
        group_df = df[mask]
        y_true = group_df['label'].values
        y_pred = group_df['pred'].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        fp_rate = fp / (fp + tn + 1e-8)  # add epsilon to avoid divide-by-zero
        fp_rates[tuple(group)] = fp_rate

    return fp_rates


fp_rates = compute_fp_rates(binary_data_test, gerry_binary_preds)
for group, rate in fp_rates.items():
    print(f"Group {group}: FP rate = {rate:.20f}")

max_diff = max(fp_rates.values()) - min(fp_rates.values())
print(f"Max FP rate disparity: {max_diff:.20f}")

#assert np.array_equal(np.unique(gerry_binary_preds), [0, 1]), "Predictions must be binary."


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
        y_true = (y_true > 0.37).astype(int)
        
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

clf.fit(aif_data_train)
preds = clf.predict(aif_data_train).labels.flatten()

from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
auditor = Auditor(aif_data_train, 'FP')

baseline_fp = auditor.get_baseline(binary_data_test.labels.flatten(), gerry_preds)
print("Baseline_fp should be:", baseline_fp)


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

gamma_list = [0.01, 0.02, 0.05, 0.00001]
pareto_iters = 100
fp_vs_fn(binary_data_test, gamma_list, pareto_iters)
img_fn='gerryfair_fp_fn.png'
image = Image.open(img_fn)
image.show()
''''
POST PROCESSING

'''
from multicalibration import MulticalibrationPredictor


probs = expit(model.predict(X_test).flatten())     # CNN logits → probabilities 
labels = (y_test.flatten() > 0.37).astype(int)      # binary ground-truth
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