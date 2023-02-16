from keras.models import load_model
import random
import json
import os
import time
import numpy as np
from keras.utils import to_categorical

INPUT_MODEL = "./Coflow_model.h5"
INPUT_MINMAX = "./min_max.json"
INPUT_FLOW = "./JSON/1_10_M1000_50_V10_1_420_delay/coflow.json"

with open(INPUT_FLOW, 'r', encoding='utf-8') as f:
    coflowData = json.load(f)

print(len(coflowData[0]['2265']))

# coflow_queue = []
# threhold = 0
# for i in range(10):
#     classify_flow = normalize(flows1[i], min_data, max_data)
#     sameScore = []
#     diffScore = []
#     for j in range(len(coflow_queue)):
#         sameScore.append(0)
#         diffScore.append(0)
#         count = 0
#         sampleNum = min(len(coflow_queue[j]), 20)
#         sampleList = sample(coflow_queue[j], sampleNum)
#         for f in sampleList: # each flow in job
#             tmp_n =  self.__normalize(selectFlows1[i], f)

