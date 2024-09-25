
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score, roc_curve, auc
import sys
import numpy as np

import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
from DSADeepFM import DeepFM
# import config as config
import joblib
import logging
import matplotlib.pyplot as plt


logging.getLogger('tensorflow').setLevel(logging.ERROR)

gpu = 0
fold = 1
epoch = 128
embedding_size = 512
deep_layers_fm = [512, 1024]

deep = f'{deep_layers_fm[0]}_{deep_layers_fm[1]}'
path = f'DSADeepFM/GPU_{gpu}/Fold{fold}_GPU/Epoch_{epoch}/Embedding_{embedding_size}/deep_{deep}/'
if not os.path.exists(path):
    os.makedirs(path)

path_best_model = path + '/Best_model/'
if not os.path.exists(path_best_model):
    os.makedirs(path_best_model)

# ------------------------------------------------------ Dataset ------------------------------------------------------#
# 读取Dataset
Xi_train = np.load('Oneil_Xi_train_fold_' + str(fold) + '.npy', mmap_mode='r+')
Xv_train = np.load('Oneil_Xv_train_fold_' + str(fold) + '.npy', mmap_mode='r+')
Xi_valid = np.load('Oneil_Xi_valid_fold_' + str(fold) + '.npy', mmap_mode='r+')
Xv_valid = np.load('Oneil_Xv_valid_fold_' + str(fold) + '.npy', mmap_mode='r+')

print('-------------------------------- Finish Xi_train, Xv_train, Xi_valid, Xv_valid --------------------------------')

y_train = np.load('Oneil_y_train_fold_' + str(fold) + '.npy', mmap_mode='r+')
y_valid = np.load('Oneil_y_valid_fold_' + str(fold) + '.npy', mmap_mode='r+')
test_y = np.load('Oneil_test_y_fold_' + str(fold) + '.npy', mmap_mode='r+')

print('-------------------------------- Finish y ---------------------------------------------------------------------')

num_train = np.load('Oneil_num_train_fold_' + str(fold) + '.npy', mmap_mode='r+')
num_valid = np.load('Oneil_num_valid_fold_' + str(fold) + '.npy', mmap_mode='r+')

print('-------------------------------- Finish num_train, num_valid --------------------------------------------------')
print('-------------------------------- Data Finish!------------------------------------------------------------------')



# ------------------------------------------------------- 超参数 -------------------------------------------------------#
# 超参数设置
# ------------------ DeepFM Model ------------------
# params
dfm_params = {
              "use_fm": True,
              "use_deep": True,
              "use_attention": True,

              "use_bn": True,
              "batch_norm_decay": 0.995,

              "embedding_size": embedding_size,
              "cate_feature_size": 72,  # 1506
              "field_size": 3,
              "numeric_feature_size": 4710,

              "dropout_fm": [1, 1],
              "dropout_deep": [0.5, 0.5, 0.5],
              "deep_layers_fm": deep_layers_fm,
              "deep_layers_activation": tf.nn.relu,

              "initial_rate": 0.001,
              "decay_rate": 0.9,

              "epoch": epoch,  # 128
              "batch_size": 128,  # 32
              "optimizer_type": "adam",
              "verbose": True,
              "l2_reg": 0.01,
              "random_seed": 2023,
              "path": path,
}


# ----------------------------------------------------- DeepFM 训练 ----------------------------------------------------#
# 模型训练
print('-------------------------------- Deep FM ----------------------------------------------------------------------')
dfm = DeepFM(**dfm_params)
ACC_list = dfm.fit(Xi_train, Xv_train, num_train, y_train, Xi_valid, Xv_valid, num_valid, y_valid)


def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

if ACC_list == 'restart':
    print(ACC_list)
    restart_program()

joblib.dump(ACC_list, (path + 'ACC_valid_epoch_fold_' + str(fold) + '.pkl'))

with open(path + 'loss_epoch.txt', 'r') as file:
    Loss = [float(line.strip()) for line in file]
joblib.dump(Loss, (path + 'Loss_train_batch_fold_' + str(fold) + '.pkl'))

# 模型预测
pre_prob = dfm.predict(Xi_valid, Xv_valid, num_valid)
joblib.dump(pre_prob, (path + 'Predict_prob_fold_' + str(fold) + '.pkl'))

threshold = 0.5
pre_binary = (pre_prob > threshold).astype(int)
joblib.dump(pre_binary, (path + 'Predict_binary_fold_' + str(fold) + '.pkl'))

test_y = test_y.astype(int)

# 计算混淆矩阵
conf_matrix = confusion_matrix(test_y, pre_binary)

# 计算准确率
accuracy = accuracy_score(test_y, pre_binary)

# 计算 AUC
roc_auc = roc_auc_score(test_y, pre_prob)

# 计算 Precision、Recall、F1
precision = precision_score(test_y, pre_binary)
recall = recall_score(test_y, pre_binary)
f1 = f1_score(test_y, pre_binary)

# 计算 PR AUC
pr_auc = average_precision_score(test_y, pre_prob)

# 计算 Kappa
total_agreements = np.sum(np.diag(conf_matrix))
total_instances = np.sum(conf_matrix)
p0 = total_agreements / total_instances
pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_instances ** 2)
kappa = (p0 - pe) / (1 - pe)

# 计算 BACC（平衡准确率）
sensitivity = recall
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
bacc = (sensitivity + specificity) / 2

# 输出结果
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("AUC:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("PR AUC:", pr_auc)
print("Kappa:", kappa)
print("Balanced Accuracy (BACC):", bacc)

# 保存结果
results = pd.DataFrame(
    {'Accuracy': [accuracy], 'AUC': [roc_auc], 'Precision': [precision], 'Recall': [recall], 'F1': [f1],
     'PR AUC': [pr_auc], 'Kappa': [kappa], 'Balanced Accuracy (BACC)': [bacc]})

results.to_csv(path + 'Evaluation_results_fold_' + str(fold) + '.csv', index=False)

# ROC curve
fpr, tpr, thresholds = roc_curve(test_y, pre_prob)
roc_auc = auc(fpr, tpr)

# Save data to a pickle file
data_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}
joblib.dump(data_dict, path + 'ROC_curve_data_fold_' + str(fold) + '.pkl')

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(path + 'ROC_fold_' + str(fold) + '.png')
plt.close()

# Plot and save the loss curve
plt.plot(Loss)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Training Loss Curve')
plt.savefig(path + 'Loss_fold_' + str(fold) + '.png')
plt.close()

print(path)