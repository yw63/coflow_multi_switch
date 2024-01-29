import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json
import ast
import networkx as nx
from networkx.algorithms import bipartite

barWidth = 0.25
'''
#avg
local_priority = [91.6, 87.4, 79.96, 80.49]
mapping_priority = [90.96, 87.6579, 87.5686667, 89.078]
diff = []
for i in range(4):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(4)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Alpha') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.5', '0.6', '0.7', '0.8']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Avg)')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_avg.png')
'''
'''
#top k
local_priority = [73.6216, 74.09, 65.37, 75.0755]
mapping_priority = [71.5, 74.569, 69.03, 85.28]
diff = []
for i in range(4):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(4)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Alpha') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.5', '0.6', '0.7', '0.8']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k)')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk.png')
'''

#coflow size
local_priority = [41.6985, 39.42, 36.0656, 29.69]
mapping_priority = [13.179, 11.2328, 15.3266, 16.9]
diff = []
for i in range(4):
    if local_priority[i]-mapping_priority[i]>0:
        diff.append(local_priority[i]-mapping_priority[i])
    else:
        diff.append(0)
x = np.arange(4)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local size estimation difference') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping size estimation difference')
plt.plot(x, diff, 'go-', label='Size estimation improvement')
plt.xlabel('Alpha') 
plt.ylabel('Size difference to global groundtruth (%)') 
plt.xticks(x, ['0.5', '0.6', '0.7', '0.8']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Coflow size estimation difference to global groundtruth (%)')
plt.legend(loc='upper right')
plt.savefig('plt/coflow_size_estimation.png')