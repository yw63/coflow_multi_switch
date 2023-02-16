import numpy as np
import csv
import os
import random
import json
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model
import sys
import time
import math

# Parameter
INPUT_PATH = "./CSV/"
INPUT_FBFILE_PATH = "./FB2010-1Hr-150-0.txt"
INPUT_MODEL = "./Coflow_model_select.h5"
INPUT_MINMAX = "./min_max.json"
OUTPUT_CSV = "./P4_RECORD/priority_table_size.csv"
OUTPUT_ACCURACY = "./P4_RECORD/classify_record.csv"
OUTPUT_COMPLETION_TIME = "./P4_RECORD/coflow_completion_time.csv"


with open(INPUT_MINMAX) as file_object:
    min_max = json.load(file_object)
    min_data = np.array(min_max['min_num'])
    max_data = np.array(min_max['max_num'])

MODEL = load_model(INPUT_MODEL)
COFLOW_NUMBER = 10
# FLOW_NUMBER = 10000 
CONTROLLER_UPDATE_TIME = 30
SKETCH_DEPTH = 3
PACKET_CNT_THRESHOLD = 20
INITIAL_TTL = 10000
INIT_QUEUE_LIMIT = 1048576.0 * 10
JOB_SIZE_MULT = 10.0
NUM_JOB_QUEUES = 10
EGRESS_RATE = 5

# Table Size
PRIORITY_TABLE_SIZE = 4096 
PACKET_CNT_TABLE_SIZE = 512 
# PACKET_CNT_TABLE_SIZE = 512 * 2
# PACKET_CNT_TABLE_SIZE = 512 * 4
# PACKET_CNT_TABLE_SIZE = 512 * 8
# PACKET_CNT_TABLE_SIZE = 512 * 16
# PACKET_CNT_TABLE_SIZE = 512 * 32
# PACKET_CNT_TABLE_SIZE = 512 * 64
FLOW_SIZE_TABLE_SIZE = PACKET_CNT_TABLE_SIZE * 4
COFLOW_TABLE_SIZE = 10

# Data Set
fb_data = {}
fb_coflow_size = {}
fb_coflow_priority = {}

# Queue
coflow_queue = {} # Coflow_ID(key), [Flows_List, Real_Coflow_ID]
input_queue = []
output_queue = []
wait_queue = []

# Table
priority_table = {} # (Match Table) Flow_ID(key), Priority
packet_count_table = [[0 for i in range(PACKET_CNT_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] # (Sketch) Packet_Count
flow_size_table = [[0 for i in range(FLOW_SIZE_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] # (Sketch) Packet_Count
flow_record_table = {} # (in Controller) Flow_ID(key), Coflow_ID, Priority, Size, TTL, Arrival_Time, Size_m, Finish
coflow_priority_table = {} # (in Controller) Coflow_ID(key), Coflow_Size, Priority
# Other
counter = 0
DNN_counter = 0
DNN_right = 0
sketch_flow_size = {}
sketch_cnt_err = 0
sketch_size_err = 0
sketch_mean_err = 0
sketch_counter = 0
priority_table_time = []
priority_table_size = []
packet_collision = [[[] for i in range(PACKET_CNT_TABLE_SIZE)] for j in range(SKETCH_DEPTH)]
flow_collision = [[[] for i in range(FLOW_SIZE_TABLE_SIZE)] for j in range(SKETCH_DEPTH)]
pkt_collision_counter = 0
flow_collision_counter = 0 
coflow_completion = {} # Coflow ID(key), Start Time, Completion Time, Duration Time, Coflow Size, Coflow Priority
with open(OUTPUT_ACCURACY, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Flow ID", "Real Coflow ID", "Classified Coflow ID", "Real Size", "Classified Size", "Real Priority", "Classified Priority"])

def readDataSet():
    global fb_data, fb_coflow_size, fb_coflow_priority
    
    def getPriority(size):
        tmp = INIT_QUEUE_LIMIT
        p = 0
        while size > tmp:
            p += 1
            tmp *= JOB_SIZE_MULT
            if p >= NUM_JOB_QUEUES:
                break
        return p

    with open(INPUT_FBFILE_PATH, "r") as f:
        first = True
        for line in f:
            if first == True:
                if first:
                    first = False
                    continue
            line = line.replace('\n', '').split(' ')
            coflow = float(line[0])
            mapper_list = []
            reducer_list = []
            size_list = []
            mapper_num = int(line[2])
            for m in range(mapper_num):
                mapper_list.append(float(line[3+m]))
            reducer_num = int(line[2+int(line[2])+1])
            for r in range(reducer_num):
                reducer_list.append(float(line[2+int(line[2])+1+r+1].split(':')[0]))
                size_list.append(float(line[2+int(line[2])+1+r+1].split(':')[1])) # MB
            fb_coflow_size[str(coflow)] = sum(size_list) * 1024 * 1024
            fb_coflow_priority[str(coflow)] = getPriority(fb_coflow_size[str(coflow)])
            for m in range(mapper_num):
                for r in range(reducer_num):
                    key = str(coflow) + "-" + str(mapper_list[m]) + "-" + str(reducer_list[r])
                    fb_data[key] = size_list[r] / mapper_num * 1024 ###
    # print(fb_data)
    # print(fb_coflow_priority)
    # print(fb_coflow_size)

def loadcsvpartial():
    def sortDir(s):
        return int(s.split("_")[0])
    input_data = []
    input_data_flow = {}
    f_cnt = 0
    c_list = []
    """"
    csv_dir = ["./CSV/1_30_M1000_50_V10_1_420_delay",
                "./CSV/31_50_M1000_50_V10_1_420_delay",
                "./CSV/51_70_M1000_50_V10_1_420_delay",
                "./CSV/71_90_M1000_50_V10_1_420_delay",
                "./CSV/91_120_M1000_50_V10_1_420_delay"]
    """
    csv_dir = sorted(os.listdir(INPUT_PATH), key=sortDir) # Sort
    count = 0
    for f1 in csv_dir: # Packet dir
        count += 1
        if count > 5:
            break
        print("open ", f1)
        for f2 in sorted(os.listdir(os.path.join(INPUT_PATH, f1))): # Host file
            print(f2, end=" ")
            data = np.loadtxt(os.path.join(os.path.join(INPUT_PATH, f1), f2), dtype=float, delimiter=",", skiprows=1, usecols=range(8))
            for i in range(len(data)): # Packets
                c_id = data[i][0] #coflow id
                m_id = data[i][3] #mapper id
                r_id = data[i][4] #reducer id
                key = str(c_id) + "-" + str(m_id) + "-" + str(r_id)
                if c_id not in c_list:
                    if len(c_list) >= COFLOW_NUMBER:
                        continue
                    c_list.append(c_id)
                if key not in input_data_flow.keys():
                    if key not in fb_data.keys() or data[i][7] == 0:
                        continue
                    # if f_cnt >= FLOW_NUMBER:
                    #     continue
                    f_cnt += 1
                    input_data_flow[key] = []
                input_data_flow[key].append(data[i])
            for key in input_data_flow.keys():
                num = (fb_data[key] / input_data_flow[key][0][7])
                if len(input_data_flow[key]) < num:
                    orgin_len = len(input_data_flow[key])
                    while len(input_data_flow[key]) < num:
                        tmp_input = input_data_flow[key].copy()
                        for d in tmp_input:
                            if len(input_data_flow[key]) < num:
                                input_data_flow[key].append(d)
                            else:
                                break
                    add = 0
                    inter = random.sample([1, 2, 3, 4, 6, 7, 8], 1)[0]
                    for i in range(len(input_data_flow[key])-orgin_len):
                        if (orgin_len + i) % inter == 0:
                            add += 1
                        input_data_flow[key][orgin_len + i][5] += add
        print("")
        # if f_cnt >= FLOW_NUMBER:
        #     break
        if len(c_list) >= COFLOW_NUMBER:
            break
    for key in input_data_flow.keys():
        for d in input_data_flow[key]:
            input_data.append(d)
    input_data = sorted(input_data, key=lambda s:s[5])
    f_id_list = input_data_flow.keys()
    return input_data, input_data_flow, f_id_list, c_list

def loadCsvData():
    def sortDir(s):
        return int(s.split("_")[0])
    input_data = []
    input_data_flow = {}
    f_cnt = 0
    c_list = []
    csv_dir = sorted(os.listdir(INPUT_PATH), key=sortDir) # Sort
    for f1 in csv_dir: # Packet dir
        print("open ", f1)
        for f2 in sorted(os.listdir(os.path.join(INPUT_PATH, f1))): # Host file
            print(f2, end=" ")
            data = np.loadtxt(os.path.join(os.path.join(INPUT_PATH, f1), f2), dtype=float, delimiter=",", skiprows=1, usecols=range(8))
            for i in range(len(data)): # Packets
                c_id = data[i][0] #coflow id
                m_id = data[i][3] #mapper id
                r_id = data[i][4] #reducer id
                key = str(c_id) + "-" + str(m_id) + "-" + str(r_id)
                if c_id not in c_list:
                    if len(c_list) >= COFLOW_NUMBER:
                        continue
                    c_list.append(c_id)
                if key not in input_data_flow.keys():
                    if key not in fb_data.keys() or data[i][7] == 0:
                        continue
                    # if f_cnt >= FLOW_NUMBER:
                    #     continue
                    f_cnt += 1
                    input_data_flow[key] = []
                input_data_flow[key].append(data[i])
            for key in input_data_flow.keys():
                num = (fb_data[key] / input_data_flow[key][0][7])
                if len(input_data_flow[key]) < num:
                    orgin_len = len(input_data_flow[key])
                    while len(input_data_flow[key]) < num:
                        tmp_input = input_data_flow[key].copy()
                        for d in tmp_input:
                            if len(input_data_flow[key]) < num:
                                input_data_flow[key].append(d)
                            else:
                                break
                    add = 0
                    inter = random.sample([1, 2, 3, 4, 6, 7, 8], 1)[0]
                    for i in range(len(input_data_flow[key])-orgin_len):
                        if (orgin_len + i) % inter == 0:
                            add += 1
                        input_data_flow[key][orgin_len + i][5] += add
        print("")
        # if f_cnt >= FLOW_NUMBER:
        #     break
        if len(c_list) >= COFLOW_NUMBER:
            break
    for key in input_data_flow.keys():
        for d in input_data_flow[key]:
            input_data.append(d)
    input_data = sorted(input_data, key=lambda s:s[5])
    f_id_list = input_data_flow.keys()
    return input_data, input_data_flow, f_id_list, c_list

def getFlowID(packet, f_id_list):
    c_id = packet[0]
    m_id = packet[3]
    r_id = packet[4]
    key = str(c_id) + "-" + str(m_id) + "-" + str(r_id)
    return list(f_id_list).index(key)

def hash(key, width, depth):
    h = (key+(depth+1)**(depth)) % width
    return h

def sketchAction(f_id, table, add_value, clear=False):
    global packet_collision, flow_collision, pkt_collision_counter, flow_collision_counter
    get_value = []
    for i in range(SKETCH_DEPTH):
        key = hash(f_id, len(table[i]), i)
        table[i][key] += add_value
        get_value.append(table[i][key])
        if clear:
            table[i][key] = 0
        # ------ Record ------
        if table == packet_count_table: # Add packet count
            if add_value != 0:
                if f_id not in packet_collision[i][key]:
                    if packet_collision[i][key] != []:
                        pkt_collision_counter += 1
                        print("Packet Count Collision - table ", i, ": ", f_id, " and ", packet_collision[i][key])
                    packet_collision[i][key].append(f_id)
            if clear == True:
                # print(packet_collision[i][key])
                if f_id in packet_collision[i][key]:
                    packet_collision[i][key].remove(f_id)
                # print("Clear key in packte size: ", f_id)
        elif table == flow_size_table: # Add flow size
            if add_value != 0:
                if f_id not in flow_collision[i][key]:
                    if flow_collision[i][key] != []:
                        flow_collision_counter += 1
                        print("Flow Size Collision - table ", i, ": ", f_id, " and ", flow_collision[i][key])
                    flow_collision[i][key].append(f_id)                        
            if clear == True:
                # print(flow_collision[i][key])
                if f_id in flow_collision[i][key]:
                    flow_collision[i][key].remove(f_id)
                # print("Clear key in flow size: ", f_id)
        # ------ Record ------
    return min(get_value)

def checkPriorityTable(f_id, packet):
    find = False
    if f_id in priority_table.keys():
        packet.append(priority_table[f_id]) # Add priority
        find = True
    else:
        packet.append(0) # Add highest priority
    return find, packet

def updatePacketCntTable(f_id, packet):
    cnt = sketchAction(f_id, packet_count_table, 1, False)
    if cnt == 1 or cnt == PACKET_CNT_THRESHOLD: 
        return True
    else:
        return False

def updateFlowSizeTable(f_id, packet):
    global sketch_flow_size
    size = sketchAction(f_id, flow_size_table, packet[6], False)
    # Record
    if f_id not in sketch_flow_size.keys():
        sketch_flow_size[f_id] = []
    if len(sketch_flow_size[f_id]) < PACKET_CNT_THRESHOLD:
        sketch_flow_size[f_id].append(packet[6])
    # Record
    return size

def classify(f_id, packet, packet_m, arrival_t):
    global DNN_counter, DNN_right
    def normalize(f_id2, packet_m, arrival_t):
        feature_time = abs(arrival_t - flow_record_table[f_id2][4]) / (max_data[0]-min_data[0])
        normalize_packet1 = (packet_m - min_data[1]) / (max_data[1] - min_data[1])
        normalize_packet2 = (flow_record_table[f_id2][5] - min_data[1]) / (max_data[1] - min_data[1])
        return np.array([[feature_time, normalize_packet1, normalize_packet2]])
    if len(coflow_queue.keys()) == 0: # Create a new queue
        return packet[0] # Real coflow ID
    sameScore = []
    diffScore = []
    sorted_coflow_keys = sorted(coflow_queue.keys())
    for i in range(len(sorted_coflow_keys)):
        sameScore.append(0)
        diffScore.append(0)
        cnt = 0
        sampleNum = min(len(coflow_queue[sorted_coflow_keys[i]][0]), 20)
        sampleList = random.sample(range(len(coflow_queue[sorted_coflow_keys[i]][0])), sampleNum)
        for j in sampleList: # Each flow in coflow
            if coflow_queue[sorted_coflow_keys[i]][0][j] not in flow_record_table.keys():
                continue
            n = normalize(coflow_queue[sorted_coflow_keys[i]][0][j], packet_m, arrival_t)
            predict_prob = MODEL.predict(n)
            predict_classes = predict_prob[0]
            sameScore[i] += predict_classes[1]
            diffScore[i] += predict_classes[0]
            cnt += 1
            # ------ Record ------
            DNN_counter += 1
            if packet[0] == coflow_queue[sorted_coflow_keys[i]][1][j] and predict_classes[1] > predict_classes[0]:
                DNN_right += 1
            if packet[0] != coflow_queue[sorted_coflow_keys[i]][1][j] and predict_classes[1] <= predict_classes[0]:
                DNN_right += 1
            # ------ Record ------
        if cnt > 0:
            sameScore[i] /= cnt
            diffScore[i] /= cnt
    score = [-1, -1] # [c_id, Max Score]
    for i in range(len(sorted_coflow_keys)):
        if sameScore[i] > diffScore[i]:
            if sameScore[i] > score[1]:
                score[1] = sameScore[i]
                score[0] = sorted_coflow_keys[i]
    if score[1] == -1: # No friend and create a new job
        if len(coflow_queue.keys()) < COFLOW_TABLE_SIZE:
            c_id = packet[0] # Real coflow ID
        else:
            # Find the smallest coflow
            small = [0, sys.maxsize] # [c_id, #]
            for i in range(len(sorted_coflow_keys)):
                if len(coflow_queue[sorted_coflow_keys[i]][0]) < small[1]:
                    small[0] = sorted_coflow_keys[i]
                    small[1] = len(coflow_queue[sorted_coflow_keys[i]][0])
            c_id = small[0] # Smallest coflow ID
    else:
        c_id = score[0] # Existing coflow ID
    return c_id

def schedule(table):
    for c_id in table.keys():
        size = table[c_id][0] * 1024 ##
        curP = 0
        tmp = INIT_QUEUE_LIMIT
        while size > tmp:
            curP += 1
            tmp *= JOB_SIZE_MULT
            if curP >= NUM_JOB_QUEUES:
                break
        table[c_id][1] = curP
    return table

def updateFlowRecordTable(f_id, packet):
    global sketch_size_err, sketch_cnt_err, sketch_mean_err, sketch_counter
    # Get data from Packet Table
    cnt = sketchAction(f_id, packet_count_table, 0, False)
    size = sketchAction(f_id, flow_size_table, 0, False)
    if cnt == 1:
        print("Put ", f_id, "in Flow Table")
        if f_id in flow_record_table.keys():
            print("(cnt = 1) Flow ", f_id, " is in Flow Table")
            flow_record_table[f_id][6] = False
            flow_record_table[f_id][3] = INITIAL_TTL
        else:
            flow_record_table[f_id] = [None, 0, size, INITIAL_TTL, packet[5], 0, False]
        return
    elif cnt == PACKET_CNT_THRESHOLD:
        sketchAction(f_id, packet_count_table, 0, True) # Reset
        # Classify
        packet_m = size / cnt
        # ------ Record ------
        if f_id in sketch_flow_size.keys(): 
            real_packet_s = sum(sketch_flow_size[f_id])
            real_packet_c = len(sketch_flow_size[f_id])
            real_packet_m = real_packet_s/real_packet_c
            if math.isnan(abs(real_packet_s - size) / real_packet_s) == False and math.isnan(abs(real_packet_c - cnt) / real_packet_c) == False and math.isnan(abs(real_packet_m - packet_m) / (real_packet_s/real_packet_c)) == False:
                sketch_size_err += abs(real_packet_s - size) / real_packet_s
                sketch_cnt_err += abs(real_packet_c - cnt) / real_packet_c
                sketch_mean_err += abs(real_packet_m - packet_m) / (real_packet_s/real_packet_c)
                sketch_counter += 1
                print("------ ", f_id, " ------")
                print("sketch size: ", size, " real size: ", real_packet_s)
                print("sketch cnt: ", cnt, " real cnt: ", real_packet_c)
                print("sketch mean: ", packet_m, " real mean: ", real_packet_s/real_packet_c)
                print("-----------------")
        # ------ Record ------
        if f_id not in flow_record_table.keys():
            print("(cnt = ", PACKET_CNT_THRESHOLD, ") Flow ", f_id, " is not in Flow Table")
            flow_record_table[f_id] = [None, 0, size, INITIAL_TTL, packet[5], packet_m, False]
        else:
            flow_record_table[f_id][2] = size
            flow_record_table[f_id][5] = packet_m
        arrival_t = flow_record_table[f_id][4]
        c_id = classify(f_id, packet, packet_m, arrival_t)
        # ------ Record ------
        real_coflow_id = packet[0]
        real_size = fb_coflow_size[str(real_coflow_id)]
        classified_size = fb_coflow_size[str(float(c_id))]
        real_priority = fb_coflow_priority[str(real_coflow_id)]
        classified_priority = fb_coflow_priority[str(float(c_id))]
        with open(OUTPUT_ACCURACY, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f_id, real_coflow_id, c_id, real_size, classified_size, real_priority, classified_priority])
        # ------ Record ------
        # Update Coflow Data and set priority
        priority = 0
        if c_id in coflow_queue.keys(): # Record in Coflow Queue for classify
            coflow_queue[c_id][0].append(f_id)
            coflow_queue[c_id][1].append(packet[0]) # Real coflow id of this flow 
        else:
            coflow_queue[c_id] = [[f_id],[packet[0]]]
        if c_id in coflow_priority_table.keys(): # Update priority
            priority = coflow_priority_table[c_id][1]
        else: # New coflow
            coflow_priority_table[c_id] = [0, 0]
        # Update Flow Table
        flow_record_table[f_id][0] = c_id
        flow_record_table[f_id][1] = priority
        # Insert to Priority Table
        if len(priority_table) < PRIORITY_TABLE_SIZE:
            priority_table[f_id] = priority
        else:
            print("(Priority Table) Overflow")
            # Todo

def PIFO(packet, wait_queue):
    wait_queue.append(packet)
    wait_queue = sorted(wait_queue, key= lambda s: s[-1], reverse=True)
    return wait_queue

def controllerUpdate(coflow_priority_table):
    # Update Flow Table (Size)
    coflow_size = {} # Coflow size, Finish 
    for f_id in flow_record_table.keys():
        if flow_record_table[f_id][6] == False: 
            size = sketchAction(f_id, flow_size_table, 0, False)
            flow_record_table[f_id][2] = size
        # For next step
        if flow_record_table[f_id][0] != None:
            if flow_record_table[f_id][0] not in coflow_size.keys():
                coflow_size[flow_record_table[f_id][0]] = flow_record_table[f_id][2]
            else:
                coflow_size[flow_record_table[f_id][0]] += flow_record_table[f_id][2]  
    # Update coflow size
    for c_id in coflow_priority_table.keys():
        if c_id not in coflow_size.keys(): # Bug
            continue
        coflow_priority_table[c_id][0] = coflow_size[c_id] 
    # Schedule
    coflow_priority_table = schedule(coflow_priority_table) # Update coflow priority
    # print("Coflow Table", coflow_priority_table)
    # Update Flow Table (Priority)
    update_flow_list = [] # Flow ID, Priority
    for f_id in flow_record_table.keys():
        if flow_record_table[f_id][0] != None: # Classified
            if flow_record_table[f_id][1] != coflow_priority_table[flow_record_table[f_id][0]][1]: # Update priority
                flow_record_table[f_id][1] = coflow_priority_table[flow_record_table[f_id][0]][1]
                update_flow_list.append([f_id, flow_record_table[f_id][1]])
    # Update Priority Table
    for entry in update_flow_list:
        if entry[0] not in priority_table.keys():
            print("(Update priority in Priority Table) Flow ", f_id, " is not in Priority Table")
            if len(priority_table) < PRIORITY_TABLE_SIZE:
                priority_table[f_id] = entry[1]
            else:
                print("(Priority Table) Overflow")
                # Todo
        else:
            priority_table[entry[0]] = entry[1]
    return coflow_priority_table 

def controllerUpdateTTL(f_id):
    clear_now = []
    finished_coflow = {}
    # Update TTL
    for f in flow_record_table.keys(): 
        if f == f_id:
            flow_record_table[f_id][3] = INITIAL_TTL
            flow_record_table[f][6] = False
        else:
            flow_record_table[f][3] -= 1
            if flow_record_table[f][3] <= 0 and flow_record_table[f][6] == False:
                print(counter, " ############### Clear", f_id)
                if flow_record_table[f][0] == None: 
                    sketchAction(f, packet_count_table, 0, True)
                    clear_now.append(f)
                else: # Classified
                    if f in priority_table.keys():
                        del priority_table[f]
                        sketchAction(f, flow_size_table, 0, True)
                    flow_record_table[f][6] = True
        if flow_record_table[f][0] != None:
            if flow_record_table[f][0] not in finished_coflow.keys():
                finished_coflow[flow_record_table[f][0]] = True
            if flow_record_table[f][6] == False: # Flow unfinished
                finished_coflow[flow_record_table[f][0]] = False # Coflow unfinished
    # Delete finished coflows
    for c_id in finished_coflow.keys(): 
        if finished_coflow[c_id] == True:
            del coflow_priority_table[c_id]
            for f in set(coflow_queue[c_id][0]):
                if f in flow_record_table.keys():
                    del flow_record_table[f]
            del coflow_queue[c_id]
    # Delete finished flows       
    for f in clear_now:
        del flow_record_table[f]
    return 

def egress(wait_queue, output_queue):
    out_packet = wait_queue.pop()
    output_queue.append(out_packet)
    # ------ Record ------
    # Coflow ID(key), Start Time, Completion Time, Duration Time, Coflow Size, Coflow Priority
    if out_packet[0] not in coflow_completion.keys():
        coflow_completion[out_packet[0]] = [counter, counter, 0, fb_coflow_size[str(out_packet[0])], fb_coflow_priority[str(out_packet[0])]]
    else:
        coflow_completion[out_packet[0]][1] = counter
        coflow_completion[out_packet[0]][2] = counter - coflow_completion[out_packet[0]][0]
    # ------ Record ------
    return output_queue

def printTable(table):
    print("----------------")
    for i in range(len(table[0])):
        if table[0][i] != 0:
            print(i, "\t", table[0][i])
    print("----------------")

def printOutputOrder(queue):
    pre = -1
    pre_c = -1
    c = 0
    for i in range(len(queue)):
        now = queue[i][-1]
        now_c = queue[i][0]
        c += 1
        if pre != now and pre_c != now_c:
            print("coflow: ", now_c, " p: ",now,  end=",")
            print(" num:", c)
            pre = now
            pre_c = now_c
            c = 0
    return

if __name__ == "__main__":
    readDataSet()
    print("Read packets data: ")
    input_queue, input_data_flow, f_id_list, c_list = loadCsvData()
    #input_queue, input_data_flow, f_id_list, c_list = loadcsvpartial()
    print(len(c_list), " coflows, ", len(f_id_list), " flows and ", len(input_queue), " packets")
    time.sleep(3) 

    packet_index = -1
    while True:
        counter += 1 # timer
        packet_index += 1
        if packet_index < len(input_queue):
            this_packet = list(input_queue[packet_index])
            f_id = getFlowID(this_packet, f_id_list)
            # Add priority into packet header
            find, this_packet = checkPriorityTable(f_id, this_packet)
            if not find:
                # Update Packet Count Table
                action = updatePacketCntTable(f_id, this_packet)
            # Update Flow Size Table
            updateFlowSizeTable(f_id, this_packet)
            # New flow or Packet full, inform controller
            if not find and action:
                updateFlowRecordTable(f_id, this_packet)
            # Controller update
            if counter % CONTROLLER_UPDATE_TIME == 0 or packet_index == len(input_queue)-1:
                coflow_priority_table = controllerUpdate(coflow_priority_table) 
            # PIFO
            wait_queue = PIFO(this_packet, wait_queue)
            
        # Egress
        if counter % EGRESS_RATE == 0:
            output_queue = egress(wait_queue, output_queue)
        # Print Result
        if counter % 100 == 0:
            print("Time slot: ", counter)
            print("Size of Priority Table: ", len(priority_table.keys()))
            priority_table_time.append(counter)
            priority_table_size.append(len(priority_table.keys()))
            if DNN_counter != 0:
                print("DNN Accuracy: ", DNN_right / DNN_counter * 100, " %")
            if sketch_counter != 0:
                print("Sketch Count Err: ", sketch_cnt_err / sketch_counter * 100, " %")
                print("Sketch Size Err: ", sketch_size_err / sketch_counter * 100, " %")
                print("Sketch Mean Err: ", sketch_mean_err / sketch_counter * 100, " %")
            print("len of wait queue: ", len(wait_queue))
        # Update TTL
        controllerUpdateTTL(f_id)
        # Completed
        if counter >= len(input_queue) and len(wait_queue) == 0: # stop
            print(len(c_list), " coflows, ", len(f_id_list), " flows and ", len(input_queue), " packets")
            print("Completed")
            print("************************************************")
            print("Accuracy: ", DNN_right / DNN_counter * 100, " %")
            print("Sketch Count Err: ", sketch_cnt_err / sketch_counter * 100, " %")
            print("Sketch Size Err: ", sketch_size_err / sketch_counter * 100, " %")
            print("Sketch Size Err: ", sketch_size_err / sketch_counter * 100, " %")
            print("Packet Count Collision: ", pkt_collision_counter)
            print("FLow Size Collision: ", flow_collision_counter)
            print("************************************************")
            # printOutputOrder(output_queue)
            break           
    
    # ------ Record ------
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["Time slot", "Size"])
        for i in range(len(priority_table_time)):
            writer.writerow([priority_table_time[i], priority_table_size[i]])
        print("Write Completed")
    with open(OUTPUT_COMPLETION_TIME, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Coflow ID", "Start Time", "Completion Time", "Duration Time", "Coflow Size", "Coflow Priority"])
        for k, v in coflow_completion.items():
            tmp = [k]
            tmp.extend(v)
            writer.writerow(tmp)
    # ------ Record ------







