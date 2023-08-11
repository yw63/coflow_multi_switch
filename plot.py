import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import json
import ast
import networkx as nx
from networkx.algorithms import bipartite

'''
filename = "Coflow_Priority_Result_2switch_30coflow_296119packets"
data = pd.read_csv("P4_RECORD/" + filename + ".csv")
timer = []
global_priority = []
local_priority = []
mapping_priority = []

for ind in data.index:
    timer.append(data["timer"][ind])
    global_tmp = data['global'][ind]
    global_tmp = global_tmp[1:-1]
    global_list = global_tmp.split(', ')
    global_priority_current = []
    for item in global_list:
        if item[0]=='(':
            global_priority_current.append(float(item[1:]))
    print("current global priority = ", global_priority_current)
    global_priority.append(global_priority_current)

    mapping_tmp = data['local with mapping'][ind]
    mapping_tmp = mapping_tmp[1:-1]
    mapping_list = mapping_tmp.split(']')
    mapping_list.pop(-1)
    mapping_priority_current = []
    for item in mapping_list:
        item = item.replace("[", "")
        print("mapping list =", item)
        tmp = item.split(', ')
        mapping_partial = []
        for i in tmp:
            if len(i)>0 and i[0]=='(':
                mapping_partial.append(float(i[1:]))
        mapping_priority_current.append(mapping_partial)
    print("current mapping priority =", mapping_priority_current)
    mapping_priority.append(mapping_priority_current)
    
    local_tmp = data['local'][ind]
    local_tmp = local_tmp[1:-1]
    local_list = local_tmp.split(']')
    local_list.pop(-1)
    local_priority_current = []
    for item in local_list:
        item = item.replace("[", "")
        print("local list =", item)
        tmp = item.split(', ')
        local_partial = []
        for i in tmp:
            if len(i)>0 and i[0]=='(':
                local_partial.append(float(i[1:]))
        local_priority_current.append(local_partial)
    print("current local priority =", local_priority_current)
    local_priority.append(local_priority_current)

print("------------------------------")
local_correctness = []
mapping_correctness = []
for i in range(len(timer)):
    correct_answer = global_priority[i]
    print("current global priority:", global_priority[i])
    print("current local priority:", local_priority[i])
    print("current mapping priority:", mapping_priority[i])

    current_local_correctness = []
    for current_local_priority in local_priority[i]:
        correct = 0
        total = 0
        if len(current_local_priority)<=1:
            current_local_correctness.append(100)
            continue
        for x in range(len(current_local_priority)):
            y = x+1
            while(y<len(current_local_priority)):
                total += 1
                if correct_answer.index(current_local_priority[x]) < correct_answer.index(current_local_priority[y]):
                    correct += 1
                y+=1
        current_local_correctness.append(correct/total*100)
    local_correctness.append(current_local_correctness)
    print("current local correctness:", current_local_correctness)

    current_mapping_correctness = []
    for current_mapping_priority in mapping_priority[i]:
        correct = 0
        total = 0
        if len(current_mapping_priority)<=1:
            current_mapping_correctness.append(100)
            continue
        for x in range(len(current_mapping_priority)):
            y = x+1
            while(y<len(current_mapping_priority)):
                total += 1
                if correct_answer.index(current_mapping_priority[x]) < correct_answer.index(current_mapping_priority[y]):
                    correct += 1
                y+=1
        current_mapping_correctness.append(correct/total*100)
    mapping_correctness.append(current_mapping_correctness)
    print("current mapping currectness:", current_mapping_correctness)
'''
'''
local_correctness_average = []
mapping_correctness_average = []
arrival_percentage = []
for i in range(len(timer)):
    print("local correctness:", local_correctness[i])
    print("mapping correctness:", mapping_correctness[i])
    local_correctness_average.append(sum(local_correctness[i])/len(local_correctness[i]))
    mapping_correctness_average.append(sum(mapping_correctness[i])/len(mapping_correctness[i]))
    arrival_percentage.append(timer[i]/timer[-1]*100)

plt.plot(arrival_percentage, local_correctness_average, 'go--')
plt.plot(arrival_percentage, mapping_correctness_average, 'ro-')
plt.legend(['avg local correctness', 'avg mapping correctness'])
plt.title('Priority Accuracy (%)')
plt.xlabel('Input Arrival Percentage (%)')
plt.ylabel('Accuracy (%)')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--')
plt.savefig("plt/" + filename + "_average.png")
'''
'''
bar = []
for i in range(len(local_correctness)):
    tmp = []
    tmp.append(timer[i])
    for correctness in local_correctness[i]:
        tmp.append(correctness)
    for correctness in mapping_correctness[i]:
        tmp.append(correctness)
    bar.append(tmp)
print("final bar:", bar)

#create data
df = pd.DataFrame(bar, columns=['Timer', 'Switch1 Local', 'Switch2 Local', 'Switch1 Mapping', 'Switch2 Mapping'])
print(df)

# plot grouped bar chart
df.plot(x='Timer',
        kind='bar',
        stacked=False,
        color = {'Switch1 Local':'red', 'Switch2 Local':'red', 'Switch1 Mapping':'blue', 'Switch2 Mapping':'blue'},
        title='Priority Accuracy')
plt.xlabel("Timer")
plt.ylabel("Accuracy (%)")
plt.savefig("plt/" + filename + ".png")
'''
'''
#top k
print("--------------top k ----------------")
local_correctness_topk = []
mapping_correctness_topk = []
k = 0.5
for i in range(len(timer)):
    correct_answer = global_priority[i]
    print("current global priority:", global_priority[i])
    print("current local priority:", local_priority[i])
    print("current mapping priority:", mapping_priority[i])

    current_local_correctness = []
    for current_local_priority in local_priority[i]:
        correct = 0
        total = 0
        
        current_local_priority = current_local_priority[int(len(current_local_priority)*k):]
        print("current local priority after paririon top k:", current_local_priority)
        if len(current_local_priority)<=1:
            current_local_correctness.append(100)
            continue
        for x in range(len(current_local_priority)):
            y = x+1
            while(y<len(current_local_priority)):
                total += 1
                if correct_answer.index(current_local_priority[x]) < correct_answer.index(current_local_priority[y]):
                    correct += 1
                y+=1
        current_local_correctness.append(correct/total*100)
    local_correctness_topk.append(current_local_correctness)
    print("current local correctness(top k):", current_local_correctness)

    current_mapping_correctness = []
    for current_mapping_priority in mapping_priority[i]:
        correct = 0
        total = 0
        current_mapping_priority = current_mapping_priority[int(len(current_mapping_priority)*k):]
        print("current mapping priority after paririon top k:", current_mapping_priority)
        if len(current_mapping_priority)<=1:
            current_mapping_correctness.append(100)
            continue
        for x in range(len(current_mapping_priority)):
            y = x+1
            while(y<len(current_mapping_priority)):
                total += 1
                if correct_answer.index(current_mapping_priority[x]) < correct_answer.index(current_mapping_priority[y]):
                    correct += 1
                y+=1
        current_mapping_correctness.append(correct/total*100)
    mapping_correctness_topk.append(current_mapping_correctness)
    print("current mapping currectness(top k):", current_mapping_correctness)

local_correctness_average_topk = []
mapping_correctness_average_topk = []
arrival_percentage = []
for i in range(len(timer)):
    print("local correctness(top k):", local_correctness_topk[i])
    print("mapping correctness(top k):", mapping_correctness_topk[i])
    local_correctness_average_topk.append(sum(local_correctness_topk[i])/len(local_correctness_topk[i]))
    mapping_correctness_average_topk.append(sum(mapping_correctness_topk[i])/len(mapping_correctness_topk[i]))
    arrival_percentage.append(timer[i]/timer[-1]*100)

plt.plot(arrival_percentage, local_correctness_average_topk, 'go--')
plt.plot(arrival_percentage, mapping_correctness_average_topk, 'ro-')
plt.legend(['avg local correctness', 'avg mapping correctness'])
plt.title('Top K Priority Accuracy (avg)')
plt.xlabel('Input Arrival Percentage (%)')
plt.ylabel('Accuracy (%)')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--')
plt.savefig("plt/" + filename + "_topk_average.png")

bar = []
for i in range(len(local_correctness_topk)):
    tmp = []
    tmp.append(timer[i])
    for correctness in local_correctness_topk[i]:
        tmp.append(correctness)
    for correctness in mapping_correctness_topk[i]:
        tmp.append(correctness)
    bar.append(tmp)
print("final bar(top k):", bar)

#create data
df = pd.DataFrame(bar, columns=['Timer', 'Switch1 Local', 'Switch2 Local', 'Switch1 Mapping', 'Switch2 Mapping'])
print(df)

# plot grouped bar chart
df.plot(x='Timer',
        kind='bar',
        stacked=False,
        color = {'Switch1 Local':'red', 'Switch2 Local':'red', 'Switch1 Mapping':'blue', 'Switch2 Mapping':'blue'},
        title='Priority Accuracy(top k)')
plt.xlabel("Timer")
plt.ylabel("Accuracy (%)")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig("plt/" + filename + "_topk.png")
'''
'''
#distance
print("--------------distance---------------")
local_distance = []
mapping_distance = []

for i in range(len(timer)):
    global_correct_answer = global_priority[i]
    print("current global priority:", global_priority[i])
    print("current local priority:", local_priority[i])
    print("current mapping priority:", mapping_priority[i])

    current_local_distance = []
    for current_local_priority in local_priority[i]:
        distance = 0
        current_local_correct_answer = []
        for cid in global_correct_answer:
            if cid in current_local_priority:
                current_local_correct_answer.append(cid)
        print("current partial local priority:", current_local_priority)
        print("correct partial local priority:", current_local_correct_answer)
        for priority in current_local_priority:
            print("index of ",priority, "in current local priority:", current_local_priority.index(priority))
            print("index of ",priority, "in correct local priority:", current_local_correct_answer.index(priority))
            distance += abs(current_local_priority.index(priority)-current_local_correct_answer.index(priority))
        current_local_distance.append(distance)
    local_distance.append(current_local_distance)    

    current_mapping_distance = []
    for current_mapping_priority in mapping_priority[i]:
        distance = 0
        current_mapping_correct_answer = []
        for cid in global_correct_answer:
            if cid in current_mapping_priority:
                current_mapping_correct_answer.append(cid)
        print("current partial mapping priority:", current_mapping_priority)
        print("correct partial mapping priority:", current_mapping_correct_answer)
        for priority in current_mapping_priority:
            print("index of ",priority, "in current mapping priority:", current_mapping_priority.index(priority))
            print("index of ",priority, "in correct mapping priority:", current_mapping_correct_answer.index(priority))
            distance += abs(current_mapping_priority.index(priority)-current_mapping_correct_answer.index(priority))
        current_mapping_distance.append(distance)
    mapping_distance.append(current_mapping_distance) 

avg_local_distance = []
avg_mapping_distance = []
arrival_percentage = []
for i in range(len(timer)):
    print("local distance:", local_distance[i])
    print("mapping distance:", mapping_distance[i])
    avg_local_distance.append(sum(local_distance[i])/len(local_distance[i]))
    avg_mapping_distance.append(sum(mapping_distance[i])/len(mapping_distance[i]))
    arrival_percentage.append(timer[i]/timer[-1]*100)
plt.plot(arrival_percentage, avg_local_distance, 'go--')
plt.plot(arrival_percentage, avg_mapping_distance, 'ro-')
plt.legend(['avg local distance', 'avg mapping distance'])
plt.title('Avg Mapping Distance')
plt.xlabel('Input Arrival Percentage (%)')
plt.ylabel('Distance to Groundtruth (lower the better)')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--')
plt.savefig("plt/" + filename + "_distance_average.png")

bar = []
for i in range(len(local_distance)):
    tmp = []
    tmp.append(timer[i])
    for distance in local_distance[i]:
        tmp.append(distance)
    for distance in mapping_distance[i]:
        tmp.append(distance)
    bar.append(tmp)
print("final bar(distance):", bar)

#create data
df = pd.DataFrame(bar, columns=['Timer', 'Switch1 Local', 'Switch2 Local', 'Switch1 Mapping', 'Switch2 Mapping'])
print(df)

# plot grouped bar chart
df.plot(x='Timer',
        kind='bar',
        stacked=False,
        color = {'Switch1 Local':'red', 'Switch2 Local':'red', 'Switch1 Mapping':'blue', 'Switch2 Mapping':'blue'},
        title='Priority Difference Distance')
plt.xlabel("Timer")
plt.ylabel("Distance")
plt.savefig("plt/" + filename + "_distance.png")
'''


#coflowsize estimation bar graph
data = pd.read_csv("P4_RECORD/CoflowSize_Estimation_Record_2switch_20coflow_203755packets_alpha=0.8.csv")
timer = []
global_size = []
mapping_size = []
local1 = []
local2 = []
for ind in data.index:
    timer.append(data["timer"][ind])
    global_size.append(float(data["global size"][ind]))
    mapping_size.append(float(data["mapping size"][ind]))
    tmp = data["local sizes"][ind][1:-1]
    tmp_list = tmp.split(',')
    item1 = float(tmp_list[0])
    item2 = float(tmp_list[1])
    local1.append(item1)
    local2.append(item2)
print("timer = ", timer)
print("global size = ", global_size)
print("mapping size = ", mapping_size)
print("local1 = ", local1)
print("local2 = ", local2)

x = np.arange(len(timer))
width = 0.2
plt.bar(x-0.3, global_size, width, color='red')
plt.bar(x-0.1, mapping_size, width, color='orange')
plt.bar(x+0.1, local1, width, color='green')
plt.bar(x+0.3, local2, width, color='cyan')
plt.xlabel("Timer")
plt.ylabel("Size")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(linestyle='--')
plt.legend(["Global size", "Mapping size", "Local size 1", "Local size 2"])
plt.title('Coflow Estimation Size (⍺=0.8)')
plt.savefig("plt/CoflowSize_Estimation_Record_2switch_20coflow_203755packets_alpha=0.8.png")

'''
#bipartite graph for mapping result
G = nx.DiGraph()

partition1 = ['A1', 'A2', 'A3', 'A4', 'A5']
partition2 = ['B1','B2', 'B3', 'B4']
G.add_nodes_from(partition1, bipartite=0)
G.add_nodes_from(partition2, bipartite=1)

edges1 = [('A4', 'B4'), ('A5', 'B4')]
edges2 = [('B2', 'A3')]
#bidirectional
edges3 = [('A1', 'B1'), ('B1', 'A1'), ('A2', 'B3'), ('B3', 'A2'), ('A3', 'B4'), ('B4', 'A3')]
G.add_edges_from(edges1, color='blue')
G.add_edges_from(edges2, color='red')
G.add_edges_from(edges3, color='green')

labels = {'A1': '8, 438, 100%', 'A2': '31, 443, 100%', 'A3': '197, 437, 100%', 'A4': '21, 437, 71%', 'A5': '55, 437, 100%',
          'B1': '11, 438, 100%', 'B2': '6, 437, 50%', 'B3': '31, 443, 94%', 'B4': '201, 437, 100%'}
nx.set_node_attributes(G, labels, 'label')

pos = {}
x = 0
y = 0

for node in partition1:
    pos[node] = (x, y)
    y += 1

x = 2
y = 0
for node in partition2:
    pos[node] = (x, y)
    y += 1

node_colors = ['skyblue' if node in partition1 else 'red' for node in G.nodes()]
edge_colors = [G.edges[edge]['color'] for edge in G.edges()]
node_labels = nx.get_node_attributes(G, 'label')

# Zoom out the graph
plt.figure(figsize=(12, 8))  # Adjust the figure size as desired
plt.axis('auto')  # Set the plot limits automatically

nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=500, font_size=12, edge_color=edge_colors, arrows=True)
legend_labels = ['First label = Queue Size', 'Second label = Major cid', 'Third label = Major percentage']
legend_colors = ['skyblue', 'lightgreen', 'red']
legend_handles = [plt.Line2D([], [], color=color, linestyle='-', linewidth=2) for color in legend_colors]
plt.legend(legend_handles, legend_labels, loc='best', handlelength=0, framealpha=1)

plt.savefig("plt/mapping_125000_175000.png", format="PNG")
'''

'''
data = pd.read_csv("P4_RECORD/classify_result_31532flows_316686packets_1switch_40coflow_0.csv")


major = []
minor1 = []
minor2 = []
score = []

counter = 5000
for ind in data.index:
    if data['Time'][ind] >= counter:
        cid = data['Classified cid'][ind]
        input_string = str(data['Queue Distribution'][ind])
        queue_distribution = ast.literal_eval(input_string)
        for key in queue_distribution.keys():
            print(key, ":", queue_distribution[key])
        value = []
        for key in queue_distribution[cid].keys():
            value.append(queue_distribution[cid][key])
        print(value)
        value = sorted(value, reverse=True)
        if len(value)>=2:
            major.append(value[0])
            minor1.append(value[1])
            counter+=5000
        
        Sum = sum(value)
        print("sum=", Sum)
        percentage = []
        for item in value:
            percentage.append(item/Sum)
        print(percentage)
        percentage = sorted(percentage, reverse=True)
        print(percentage)
        if len(percentage)>=2:
            major.append(percentage[0])
            minor1.append(percentage[1])
            if data['Score'][ind]>0:
                score.append(data['Score'][ind])
            else:
                score.append(0)
            counter+=5000
        
print("major = ", major)
print("minor1 = ", minor1)
x = np.arange(len(major))
print(x)
print(len(major), len(minor1))
plt.bar(x, major, color='b')
plt.bar(x, minor1, bottom=major, color='r')
#plt.plot(x, score, color='gold')
#plt.bar(x, minor2, bottom=minor1, color='y')
# 移除右邊和上面的邊框
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('Sample')
plt.ylabel('Flow Count')
plt.legend(["Major", "minor"])
plt.title("Coflow's flow count in classified queue")
plt.savefig('plt/coflow_flowcount_in_classifiedq_31532flows_316686packets_1switch_40coflow_0.png')
'''
'''
x1 = []
x2 = []
y1 = []
y2 = []
for ind in data.index:
    if data['Dominated'][ind]==True:
        x1.append(data['Queue Size'][ind])
        y1.append(data['Score'][ind])
    else:
        x2.append(data['Queue Size'][ind])
        y2.append(data['Score'][ind])
plt.scatter(x1,y1,marker='o', color='blue',alpha=0.5, label='Dominated')
plt.scatter(x2,y2,marker='x', color='red',alpha=0.5, label='Not dominated')
# 移除右邊和上面的邊框
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('Queue Size')
plt.ylabel('Score')
plt.legend(loc='best')
#plt.xlim([0.075,0.175])
#plt.ylim([0, 0.3])
plt.savefig('plt/qsize_to_score_dominated_31532flows_316686packets_1switch_40coflow_0.png')
'''
'''
data = pd.read_csv("P4_RECORD/classify_result_1switch_20coflow_0.csv")
y1 = []
y2 = []
for ind in data.index:
    if data['Score'][ind] == -1:
        continue
    if data['Correct'][ind] == True:
        y1.append(data['Score'][ind])
    else:
        y2.append(data['Score'][ind])
print(y1)
print(y2)
x1 = np.arange(len(y1))
x2 = np.arange(len(y2))
plt.scatter(x1,y1,color='blue')
plt.scatter(x2,y2,color='red')
plt.ylabel('Score')
plt.savefig('plt/classify.png')
'''
'''
data = pd.read_csv("P4_RECORD/classify_accuracy_2switch_20coflow_0.csv")
cid_list = []
for cid in data.iloc[0:0]:
    cid_list.append(cid)
print("cid_list: ", cid_list)
shuffle_cid = cid_list
random.shuffle(shuffle_cid)
sample_cid = shuffle_cid[0:5]
cid_major_accuracy = {}
cid_minor_accuracy = {}
for cid in sample_cid:
    cid_major_accuracy[cid] = [[],[]]
    cid_minor_accuracy[cid] = [[],[]]

data.columns=data.iloc[0]
data = data.drop(data.index[0])
data = data.drop(data.columns[6:17], axis=1)
time = np.array(data["Time"])
for i in range(len(data)):
    if data.iloc[i]["Coflow id"] in sample_cid:
        cid_major_accuracy[data.iloc[i]["Coflow id"]][0].append(int(data.iloc[i]["Time"]))
        cid_major_accuracy[data.iloc[i]["Coflow id"]][1].append(float(data.iloc[i]["Major Accuracy"]))
        cid_minor_accuracy[data.iloc[i]["Coflow id"]][0].append(int(data.iloc[i]["Time"]))
        cid_minor_accuracy[data.iloc[i]["Coflow id"]][1].append(float(data.iloc[i]["Minor Accuracy"]))

print(type(cid_major_accuracy[sample_cid[0]][1][0]))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlim([0,54000])
plt.ylim([0.0,1.2])
plt.xlabel("Time")
plt.ylabel("Accuracy")
major_color = ['brown', 'orange', 'olivedrab', 'teal', 'mediumorchid']
minor_color = ['lightcoral', 'gold', 'darkseagreen', 'darkturquoise', 'plum']
for cid in sample_cid:
    x = np.array(cid_major_accuracy[cid][0])
    y = np.array(cid_major_accuracy[cid][1])
    ci = major_color[sample_cid.index(cid)]
    plt.scatter(x, y, marker='o', color=ci)
for cid in sample_cid:
    x = np.array(cid_minor_accuracy[cid][0])
    y = np.array(cid_minor_accuracy[cid][1])
    ci = minor_color[sample_cid.index(cid)]
    plt.scatter(x, y, marker='^', color=ci)
plt.savefig("plt/dot_20coflows_accuracy_0.png")
plt.xlim([0,11000])
plt.savefig("plt/dot_20coflows_accuracy_0_partial0.png")
plt.xlim([11000,22000])
plt.savefig("plt/dot_20coflows_accuracy_0_partial1.png")
plt.xlim([22000,33000])
plt.savefig("plt/dot_20coflows_accuracy_0_partial2.png")
plt.xlim([33000,44000])
plt.savefig("plt/dot_20coflows_accuracy_0_partial3.png")
plt.xlim([44000,55000])
plt.savefig("plt/dot_20coflows_accuracy_0_partial4.png")

#2nd switch
data = pd.read_csv("P4_RECORD/classify_accuracy_2switch_20coflow_1.csv")
cid_major_accuracy = {}
cid_minor_accuracy = {}
for cid in sample_cid:
    cid_major_accuracy[cid] = [[],[]]
    cid_minor_accuracy[cid] = [[],[]]
'''
'''
data.columns=data.iloc[0]
data = data.drop(data.index[0])
data = data.drop(data.columns[6:17], axis=1)
time = np.array(data["Time"])
for i in range(len(data)):
    if data.iloc[i]["Coflow id"] in sample_cid:
        cid_major_accuracy[data.iloc[i]["Coflow id"]][0].append(int(data.iloc[i]["Time"]))
        cid_major_accuracy[data.iloc[i]["Coflow id"]][1].append(float(data.iloc[i]["Major Accuracy"]))
        cid_minor_accuracy[data.iloc[i]["Coflow id"]][0].append(int(data.iloc[i]["Time"]))
        cid_minor_accuracy[data.iloc[i]["Coflow id"]][1].append(float(data.iloc[i]["Minor Accuracy"]))

print(type(cid_major_accuracy[sample_cid[0]][1][0]))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlim([0,54000])
plt.ylim([0.0,1.2])
plt.xlabel("Time")
plt.ylabel("Accuracy")
major_color = ['brown', 'orange', 'olivedrab', 'teal', 'mediumorchid']
minor_color = ['lightcoral', 'gold', 'darkseagreen', 'darkturquoise', 'plum']
for cid in sample_cid:
    x = np.array(cid_major_accuracy[cid][0])
    y = np.array(cid_major_accuracy[cid][1])
    ci = major_color[sample_cid.index(cid)]
    plt.scatter(x, y, marker='o', color=ci)
for cid in sample_cid:
    x = np.array(cid_minor_accuracy[cid][0])
    y = np.array(cid_minor_accuracy[cid][1])
    ci = minor_color[sample_cid.index(cid)]
    plt.scatter(x, y, marker='^', color=ci)
plt.savefig("plt/dot_20coflows_accuracy_1.png")
plt.xlim([0,11000])
plt.savefig("plt/dot_20coflows_accuracy_1_partial0.png")
plt.xlim([11000,22000])
plt.savefig("plt/dot_20coflows_accuracy_1_partial1.png")
plt.xlim([22000,33000])
plt.savefig("plt/dot_20coflows_accuracy_1_partial2.png")
plt.xlim([33000,44000])
plt.savefig("plt/dot_20coflows_accuracy_1_partial3.png")
plt.xlim([44000,55000])
plt.savefig("plt/dot_20coflows_accuracy_1_partial4.png")
'''