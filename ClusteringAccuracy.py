from keras.models import load_model
import random
import json
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.cluster import KMeans
import joblib
import pandas
from sklearn.metrics import accuracy_score

INPUT_MODEL = "./Coflow_model2.h5"
INPUT_MODEL3 = "./Coflow_model3.h5"
INPUT_DT = "VFDT_model.sav"
INPUT_MINMAX = "./min_max.json"
INPUT_MINMAX2 = "./min_max2.json"
INPUT_MINMAX3 = "./min_max3.json"
INPUT_PATH_1 = "./FLOW_FIRST/301_320_M1000_50_V10_1_420_delay/"
INPUT_PATH_2 = "./FLOW_ALL/301_320_M1000_50_V10_1_420_delay/"
with open(INPUT_MINMAX) as file_object:
    min_max = json.load(file_object)
    min_data = np.array(min_max['min_num'])
    max_data = np.array(min_max['max_num'])
with open(INPUT_MINMAX) as file_object:
    min_max2 = json.load(file_object)
    min_data2 = np.array(min_max2['min_num'])
    max_data2 = np.array(min_max2['max_num'])
with open(INPUT_MINMAX) as file_object:
    min_max3 = json.load(file_object)
    min_data3 = np.array(min_max3['min_num'])
    max_data3 = np.array(min_max3['max_num'])

model = load_model(INPUT_MODEL)
tree = joblib.load(INPUT_DT)
model2 = load_model(INPUT_MODEL3)

# Accuracy
def f_score(cluster, labels):
    TP, TN, FP, FN = 0, 0, 0, 0
    n = len(labels)
    # a lookup table
    for i in range(n):
        for j in range(i + 1, n):
            same_label = (labels[i] == labels[j])
            same_cluster = (cluster[i] == cluster[j])
            if same_cluster:
                if same_label:
                    TP += 1
                else:
                    FP += 1
            elif same_label:
                FN += 1
            else:
                TN += 1
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    if TP + FP + FN + TN == 0:
        acc = -1
    else:
        acc = (TN + TP) / (TP + FP + FN + TN)
    return fscore, precision, recall, acc

def normalize(flow, min_data, max_data):
    normalize_flow = [0 for i in range(len(flow))]
    normalize_flow[0] = flow[0] # label
    normalize_flow[1] = flow[1] # start time
    normalize_flow[2:] = (flow[2:]-min_data[1:]) / (max_data[1:]-min_data[1:])
    return normalize_flow

# Read Data
flows1 = [] # test data
flows2 = [] # classified data
flows1_ = [] # test data(for DT)
flows2_ = [] # classified data(for DT)
flows1__ = [] # test data(for DNN3)
flows2__ = [] # test data(for DNN3)
data = np.array([0,0,0,0,0,0])
for file1 in os.listdir(INPUT_PATH_1): 
    path = os.path.join(INPUT_PATH_1, file1)
    tmp = np.loadtxt(path, dtype=float, delimiter=",", skiprows=1)
    np.random.shuffle(tmp)
    data = np.vstack([data, tmp[0:50]])
    for t in tmp[0:50]:
        flows1.append(normalize(t, min_data, max_data))
        flows1_.append(t[:3])
        flows1__.append(t)
    path = os.path.join(INPUT_PATH_2, file1)
    tmp = np.loadtxt(path, dtype=float, delimiter=",", skiprows=1)
    np.random.shuffle(tmp)
    data = np.vstack([data, tmp[0:100]])
    for t in tmp[0:500]:
        flows2.append(normalize(t, min_data, max_data))
        flows2_.append(t[:3])
        flows2__.append(t)

data = data[1:]

# Get Label
label = []
for d in data:
    label.append(d[0])
print("label num:", len(list(set(label))))

data = data[:, 1:]
print("data shape: ", data.shape)

# kmeans
center_num = len(list(set(label)))
kmeans = KMeans(n_clusters=center_num)
kmeans.fit(data)
result_kmeans = kmeans.predict(data)
print("kmeans")
fscore, precision, recall, acc = f_score(result_kmeans[:len(flows1)], label[:len(flows1)])
print("fscore: ", fscore)
print("precision: ", precision)
print("recall: ", recall)
print("accuracy: ", acc)
print("cluster: ", len(list(set(result_kmeans[:len(flows1)]))))

# DNN
# create old queue
cluster = {}
for f in flows2:
    if f[0] not in cluster.keys():
        cluster[f[0]] = []
        cluster[f[0]].append(f)
    else:
        cluster[f[0]].append(f)

old_queue = []
for key, value in cluster.items():
    old_queue.append(value)

cluster_label = []
real_label = []

for f in flows1:
    real_label.append(f[0])
    same_score = []
    diff_score = []
    for j in range(len(old_queue)): # each coflow queue
        same_score.append(0)
        diff_score.append(0)
        count = 0
        sampleNum = min(len(old_queue[j]), 20)
        sampleList = random.sample(old_queue[j], sampleNum)
        for sl in sampleList: # each flow in job
            time_feature = abs(f[1] - sl[1])
            time_feature_normal = time_feature / (max_data[0]-min_data[0])
            tmp = []
            tmp.append(time_feature_normal)
            tmp  = tmp + f[2:] + sl[2:]
            tmp_n = np.array([tmp])
            result = model.predict_proba(tmp_n)
            same_score[j] += result[0][1]
            diff_score[j] += result[0][0]
            count += 1
        if count > 0:
            same_score[j] /= count
            diff_score[j] /= count
    queue_score = [0, 0] # [index, score]
    for j in range(len(old_queue)):
        if same_score[j] > (diff_score[j]):
            if same_score[j] > queue_score[1]:
                queue_score[1] = same_score[j]
                queue_score[0] = j
    if queue_score[1] == 0: # no friend
        # print('create new queue')
        cluster_label.append(len(old_queue)+1)
        old_queue.append([f])
    else: # cluster
        cluster_label.append(queue_score[0])

print("DNN2")
fscore, precision, recall, acc = f_score(real_label, cluster_label)
print("fscore: ", fscore)
print("precision: ", precision)
print("recall: ", recall)
print("accuracy: ", acc)
print("cluster: ", len(list(set(cluster_label))))

# DNN3
# create old queue
cluster = {}
for f in flows2__:
    if f[0] not in cluster.keys():
        cluster[f[0]] = []
        cluster[f[0]].append(f)
    else:
        cluster[f[0]].append(f)

old_queue = []
for key, value in cluster.items():
    old_queue.append(value)

cluster_label = []
real_label = []

for f in flows1__:
    real_label.append(f[0])
    same_score = []
    diff_score = []
    for j in range(len(old_queue)): # each coflow queue
        same_score.append(0)
        diff_score.append(0)
        count = 0
        sampleNum = min(len(old_queue[j]), 20)
        sampleList = random.sample(old_queue[j], sampleNum)
        for sl in sampleList: # each flow in job
            tmp = []
            for k in range(1,len(f)):
                tmp.append(abs(f[k] - sl[k])/(max_data[k-1]-min_data[k-1]))
            tmp_n = np.array([tmp])
            result = model2.predict_proba(tmp_n)
            same_score[j] += result[0][1]
            diff_score[j] += result[0][0]
            count += 1
        if count > 0:
            same_score[j] /= count
            diff_score[j] /= count
    queue_score = [0, 0] # [index, score]
    for j in range(len(old_queue)):
        if same_score[j] > (diff_score[j]):
            if same_score[j] > queue_score[1]:
                queue_score[1] = same_score[j]
                queue_score[0] = j
    if queue_score[1] == 0: # no friend
        # print('create new queue')
        cluster_label.append(len(old_queue)+1)
        old_queue.append([f])
    else: # cluster
        cluster_label.append(queue_score[0])

print("DNN3")
fscore, precision, recall, acc = f_score(real_label, cluster_label)
print("fscore: ", fscore)
print("precision: ", precision)
print("recall: ", recall)
print("accuracy: ", acc)
print("cluster: ", len(list(set(cluster_label))))

# DT
# create old queue
cluster = {}
for f in flows2_:
    if f[0] not in cluster.keys():
        cluster[f[0]] = []
        cluster[f[0]].append(f)
    else:
        cluster[f[0]].append(f)

old_queue = []
for key, value in cluster.items():
    old_queue.append(value)

cluster_label = []
real_label = []

for f in flows1_:
    real_label.append(f[0])
    same_score = []
    diff_score = []
    for j in range(len(old_queue)): # each coflow queue
        same_score.append(0)
        diff_score.append(0)
        count = 0
        sampleNum = min(len(old_queue[j]), 20)
        sampleList = random.sample(old_queue[j], sampleNum)
        for sl in sampleList: # each flow in job
            time_feature = abs(f[1] - sl[1])
            time_feature_normal = time_feature / (max_data2[0]-min_data2[0])
            tmp_s1 = (f[2] - min_data2[1]) / (max_data2[1]-min_data2[1])
            tmp_s2 = (sl[2] - min_data2[1]) / (max_data2[1]-min_data2[1])
            size_feature = abs(tmp_s1 - tmp_s2)
            tmp = [time_feature_normal, size_feature]
            tmp_n = np.array([tmp])
            result = tree.predict(tmp_n)
            if result:
                same_score[j] += 1
            else:
                diff_score[j] += 1
            count += 1
        if count > 0:
            same_score[j] /= count
            diff_score[j] /= count
    queue_score = [0, 0] # [index, score]
    for j in range(len(old_queue)):
        if same_score[j] > (diff_score[j]):
            if same_score[j] > queue_score[1]:
                queue_score[1] = same_score[j]
                queue_score[0] = j
    if queue_score[1] == 0: # no friend
        # print('create new queue')
        cluster_label.append(len(old_queue)+1)
        old_queue.append([f])
    else: # cluster
        cluster_label.append(queue_score[0])

# print(cluster_label)

print("DT")
fscore, precision, recall, acc = f_score(real_label, cluster_label)
print("fscore: ", fscore)
print("precision: ", precision)
print("recall: ", recall)
print("accuracy: ", acc)
print("cluster: ", len(list(set(cluster_label))))


