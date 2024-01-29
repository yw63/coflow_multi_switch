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
local_priority = [76.89, 78.232, 85.87, 87.6, 88.86]
mapping_priority = [85.78, 86.836, 91.9256, 90.2, 88.86]
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
plt.xlabel('Coflow size (Smallest to largest)') 
plt.ylabel('Priority Accuracy (%)') 
plt.xticks(x, ['1', '2', '3', '4', '5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Priority Accuracy Comparison (Avg), Alpha=0.8')
plt.legend(loc='center right')
plt.savefig('plt/priority_accuracy_comparison_avg_x=coflowsize.png')
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

'''
#coflow size
local_priority = [22.08, 22.6536, 20.53, 33.027, 37.59]
mapping_priority = [11.97, 7.24455, 8.212, 18.849, 17.76]
diff = []
for i in range(5):
    if local_priority[i]-mapping_priority[i]>0:
        diff.append(local_priority[i]-mapping_priority[i])
    else:
        diff.append(0)
x = np.arange(5)
plt.bar(x-0.125, local_priority, color ='lightskyblue', width = barWidth, label ='Local size estimation difference') 
plt.bar(x+0.125, mapping_priority, color ='orange', width = barWidth, label ='Mapping size estimation difference')
plt.plot(x, diff, 'go-', label='Size estimation improvement')
plt.xlabel('Coflow size (Smallest to largest)') 
plt.ylabel('Size difference to global groundtruth (%)') 
plt.xticks(x, ['1', '2', '3', '4', '5']) 

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Coflow size estimation difference to global groundtruth (%), Alpha=0.8')
plt.legend(loc='upper right')
plt.savefig('plt/coflow_size_estimation_x=coflowSize.png')
'''