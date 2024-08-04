from collections import Counter
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression

# ================================================================================
# eGFR-UDLR with CHA data

df_urine=pd.read_csv('/home/latte/project/retina_ckd/data/table/cha/cha_egfr_dataset_with_proteinuria_protein_trace_becomes_negative.csv',encoding='utf-8')

del df_urine['patient_id']

df_train_set_patient_id=pd.read_csv('/home/latte/project/retina_ckd/evaluation_by_ymp/result_files/train_set_patient_id.csv',encoding='utf-8')

df_test_set_patient_id=pd.read_csv('/home/latte/project/retina_ckd/evaluation_by_ymp/result_files/test_set_patient_id.csv',encoding='utf-8')

joint_urine_train_raw=df_urine.merge(df_train_set_patient_id,left_on=['episode_id','protein','age','htn_def','dm_def'],right_on=['patient_id','protein','age','htn','dm'],how='inner')[['episode_id','male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte','eGFR_ckd','dm_def','htn_def','sex']]
joint_urine_test_raw=df_urine.merge(df_test_set_patient_id,left_on=['episode_id','protein','age','htn_def','dm_def'],right_on=['patient_id','protein','age','htn','dm'],how='inner')[['episode_id','male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte','eGFR_ckd','dm_def','htn_def','sex']]
joint_urine_train_raw=joint_urine_train_raw.drop_duplicates(keep='first')
joint_urine_test_raw=joint_urine_test_raw.drop_duplicates(keep='first')
urine_train_data=joint_urine_train_raw[['episode_id','male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte','eGFR_ckd']]
urine_test_data=joint_urine_test_raw[['episode_id','male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte','eGFR_ckd']]

# ================================================================================
def egfr60_labeler(x):
  if x<60:return 1
  elif x>=60:return 0

# ================================================================================
urine_train_data['eGFR60_under']=urine_train_data['eGFR_ckd'].apply(egfr60_labeler)
urine_test_data['eGFR60_under']=urine_test_data['eGFR_ckd'].apply(egfr60_labeler)

# Split data into train and test sets
X_train=urine_train_data[['male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte']]
X_test=urine_test_data[['male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte']]

y_train=urine_train_data['eGFR60_under']
y_test=urine_test_data['eGFR60_under']

# ================================================================================
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================================================================
model = LogisticRegression()

model.fit(X_train, y_train)

# ================================================================================
# Make predictions on the testing set
predictions = model.predict_proba(X_test)

# ================================================================================
# Evaluate the model
accuracy = np.mean(predictions == y_test)

# Calculate the probability estimates for positive class (class 1)
probabilities = model.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, probabilities)

# ================================================================================
d={'preds60':probabilities,
   'eGFR60':y_test}
data=pd.DataFrame(d)

boot_auc = []
np.random.seed(42)

from tqdm import tqdm
for i in tqdm(range(30000)):
    df = resample(data, n_samples=len(data), stratify=data['eGFR60'])
    
    g_truth = np.asarray(df['eGFR60'])
    predictions = np.asarray(df['preds60'])

    fpr, tpr, thresholds = roc_curve(g_truth.astype(int), predictions, pos_label=1)
    AUC = auc(fpr, tpr)
    
    boot_auc.append(round(AUC,4))

CI_auc = np.percentile(boot_auc,[2.5,97.5])

# ================================================================================
# eGFR-UDLR with SCHPC data

df_urine_schpc=pd.read_csv('/home/latte/project/retina_ckd/data/table/severance/checkupcenter/SCHPC_eGFR_v6_delete_egfr_null_delete_one_image_data_delete_two_image_id_data.csv',encoding='utf-8')[['episode_id','male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte','eGFR_ckd']]
df_urine_schpc=df_urine_schpc.drop_duplicates(keep='first')

df_external_test_set_patient_id=pd.read_csv('/home/latte/project/retina_ckd/evaluation_by_ymp/result_files/schpc_id_data.csv',encoding='utf-8')

df_external_test_set_patient_id_info=df_external_test_set_patient_id[['patient_id']]

joint_data_schpc=df_external_test_set_patient_id_info.merge(df_urine_schpc,left_on='patient_id',right_on='episode_id',how='inner')

# ================================================================================
def egfr60_labeler(x):
  if x<60:return 1
  elif x>=60:return 0

# ================================================================================
joint_data_schpc['eGFR60_under']=joint_data_schpc['eGFR_ckd'].apply(egfr60_labeler)

joint_data_schpc=joint_data_schpc.drop(columns=['patient_id','episode_id','eGFR_ckd'])

X_test=joint_data_schpc[['male','age','blood','bilirubin','urobilinogen','ketone','protein','nitrite','glucose','ph','sg','leucocyte']]
y_test=joint_data_schpc['eGFR60_under']

# ================================================================================
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

predictions = model.predict(X_test)

# ================================================================================
# Evaluate the model
accuracy = np.mean(predictions == y_test)

# Calculate the probability estimates for positive class (class 1)
probabilities = model.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, probabilities)

# ================================================================================
d={'preds60':probabilities,
   'eGFR60':y_test}
data=pd.DataFrame(d)

boot_auc = []
np.random.seed(42)

for i in range(10000):
    df = resample(data, n_samples=len(data), stratify=data['eGFR60'])
    
    g_truth = np.asarray(df['eGFR60'])
    predictions = np.asarray(df['preds60'])

    fpr, tpr, thresholds = roc_curve(g_truth.astype(int), predictions, pos_label=1)
    AUC = auc(fpr, tpr)
    
    boot_auc.append(round(AUC,4))

CI_auc = np.percentile(boot_auc,[2.5,97.5])
