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


#top k
'''
#coflow size 1
local_priority = [90.9, 82.79, 81.59, 80.45, 69.03]
mapping_priority = [92.3287, 85.024, 85.5359, 87.56,79.238]
diff = []
for i in range(5):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Top k') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k), alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk_coflowsize=1_alpha=0.8_x=topk.png')
'''
'''
#coflow size=2
local_priority = [84.8, 77.7889, 66.58, 64.75, 55.76]
mapping_priority = [84.74, 77.059, 68.169, 63.4929,57.03]
diff = []
for i in range(5):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Top k') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k), alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk_coflowsize=2_alpha=0.8_x=topk.png')
'''
'''
#coflow size=3
local_priority = [90.099, 81.04, 77.519, 81.16, 77.28]
mapping_priority = [90.099, 81.84, 78.476, 82.9, 77.367]
diff = []
for i in range(5):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Top k') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k), alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk_coflowsize=3_alpha=0.8_x=topk.png')
'''
'''
#coflow size=4
local_priority = [85.06, 74.3259, 69.14, 61.86, 52.9]
mapping_priority = [85.06, 75.26, 71.6, 63.678, 57.9]
diff = []
for i in range(5):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Top k') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k), alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk_coflowsize=4_alpha=0.8_x=topk.png')
'''
#coflow size=5
local_priority = [91.2, 86.37755, 81.68, 78.448, 70.988]
mapping_priority = [91.2, 86.37755, 81.68, 78.448, 70.988]
diff = []
for i in range(5):
    if mapping_priority[i]-local_priority[i]>0:
        diff.append(mapping_priority[i]-local_priority[i])
    else:
        diff.append(0)

x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local Priority Accuracy') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping Priority Accuracy')
plt.plot(x, diff, 'go-', label='Accuracy Improvement')
plt.xlabel('Top k') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['0.1', '0.2', '0.3', '0.4', '0.5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Top k), alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_topk_coflowsize=5_alpha=0.8_x=topk.png')