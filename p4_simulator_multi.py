import numpy as np
import csv
import os
import random
import json
from keras.models import load_model
import sys
import time
import math
from queue import PriorityQueue
from numba import jit, cuda

# Parameter
INPUT_PATH = "./CSV/"
INPUT_FBFILE_PATH = "./FB2010-1Hr-150-0.txt"
INPUT_MODEL = "./Coflow_model_select.h5"
INPUT_MINMAX = "./min_max.json"
OUTPUT_CSV = "./P4_RECORD/sampleAndLabel_priority_table_size.csv"
OUTPUT_ACCURACY = "./P4_RECORD/sampleAndLabel_classify_record.csv"
OUTPUT_COMPLETION_TIME = "./P4_RECORD/sampleAndLabel_coflow_completion_time.csv"


with open(INPUT_MINMAX) as file_object:
    min_max = json.load(file_object)
    min_data = np.array(min_max['min_num'])
    max_data = np.array(min_max['max_num'])

MODEL = load_model(INPUT_MODEL)
# COFLOW_NUMBER = 100
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

counter = 0
# Data Set
fb_data = {}
fb_coflow_size = {}
fb_coflow_priority = {}

class Switch:
    def __init__(self):
        # Queue
        self.coflow_queue = {} # Coflow_ID(key), [Flows_List, Real_Coflow_ID] #v
        self.input_queue = [] #v
        self.output_queue = []
        self.wait_queue = PriorityQueue()

        # Table
        self.priority_table = {} # (Match Table) Flow_ID(key), Priority #v
        self.packet_count_table = [[0 for i in range(PACKET_CNT_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] # (Sketch) Packet_Count #v
        self.flow_size_table = [[0 for i in range(FLOW_SIZE_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] # (Sketch) Packet_Count #v
        self.flow_record_table = {} # (in Controller) Flow_ID(key), Coflow_ID, Priority, Size, TTL, Arrival_Time, Size_m, Finish #v
        self.coflow_priority_table = {} # (in Controller) Coflow_ID(key), Coflow_Size, Priority
        # Other
        self.DNN_counter = 0
        self.DNN_right = 0
        self.sketch_flow_size = {} #v
        self.sketch_cnt_err = 0 #v
        self.sketch_size_err = 0 #v
        self.sketch_mean_err = 0 #v
        self.sketch_counter = 0 #v
        self.priority_table_time = []
        self.priority_table_size = []
        self.packet_collision = [[[] for i in range(PACKET_CNT_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] #v
        self.flow_collision = [[[] for i in range(FLOW_SIZE_TABLE_SIZE)] for j in range(SKETCH_DEPTH)] #v
        self.pkt_collision_counter = 0 #v
        self.flow_collision_counter = 0 #v
        self.coflow_completion = {} # Coflow ID(key), Start Time, Completion Time, Duration Time, Coflow Size, Coflow Priority

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
                    #if len(c_list) >= COFLOW_NUMBER:
                    #    continue
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
        # if len(c_list) >= COFLOW_NUMBER:
        #    break
    for key in input_data_flow.keys():
        for d in input_data_flow[key]:
            input_data.append(d)
    input_data = sorted(input_data, key=lambda s:s[5])
    f_id_list = input_data_flow.keys()
    return input_data, input_data_flow, f_id_list, c_list

def sampling(input_queue, input_data_flow, f_id_list, c_list, k):
    shuffle_c_list = sorted(c_list)
    random.shuffle(shuffle_c_list)
    sample_c_list = shuffle_c_list[:k]
    sample_f_id_list = []
    for cid in sample_c_list:
        for fid in f_id_list:
            if(fid.split('-',1)[0] == str(cid)):
                sample_f_id_list.append(fid)
    set_sample_fid_list = set(sample_f_id_list)
    sample_input_data_flow = input_data_flow.copy()
    for key in input_data_flow:
        if key not in set_sample_fid_list:
            sample_input_data_flow.pop(key)
    sample_input_queue = []
    for item in input_queue:
        fid = str(item[0]) + "-" + str(item[3]) + "-" + str(item[4])
        if fid in set_sample_fid_list:
            sample_input_queue.append(item)
    return sample_input_queue, sample_input_data_flow, sample_f_id_list, sample_c_list

def grouping(switches, sample_input_queue, sample_input_flow, sample_f_id_list, k):
    shuffle_fid = sample_f_id_list
    random.shuffle(shuffle_fid)
    shuffle_fid_list = np.array_split(shuffle_fid, k)
    shuffle_fid_list_sets = []
    for fid_list in shuffle_fid_list:
        shuffle_fid_list_sets.append(set(fid_list))
    switch_datas = [[]for i in range(k)]
    for item in sample_input_queue:
        key = str(item[0]) + "-" + str(item[3]) + "-" + str(item[4])
        for fid_list_set in shuffle_fid_list_sets:
            if key in fid_list_set:
                switch_datas[shuffle_fid_list_sets.index(fid_list_set)].append(item)
    for switch in switches:
        switch.input_queue = switch_datas[switches.index(switch)]
    return switches

def grouping2(switches, sample_input_queue, sample_input_flow, sample_f_id_list, sample_c_list, numOfSwitches):
    shuffle_cid = sample_c_list
    print("before shuffle: ",shuffle_cid)
    random.shuffle(shuffle_cid)
    print("after shuffle: ", shuffle_cid)
    shuffle_cid_list = np.array_split(shuffle_cid, numOfSwitches)
    shuffle_cid_list_sets = []
    for cid_list in shuffle_cid_list:
        shuffle_cid_list_sets.append(set(cid_list))
    switch_datas = [[]for i in range(numOfSwitches)]
    for item in sample_input_queue:
        cid = item[0]
        for cid_list_set in shuffle_cid_list_sets:
            if cid in cid_list_set:
                switch_datas[shuffle_cid_list_sets.index(cid_list_set)].append(item)
    for switch in switches:
        switch.input_queue = switch_datas[switches.index(switch)]
    return switches, shuffle_cid_list

def getFlowID(packet, f_id_list):
    c_id = packet[0]
    m_id = packet[3]
    r_id = packet[4]
    key = str(c_id) + "-" + str(m_id) + "-" + str(r_id)
    return list(f_id_list).index(key)

def checkPriorityTable(switch, f_id, packet):
    find = False
    if f_id in switch.priority_table.keys():
        packet.append(switch.priority_table[f_id]) # Add priority
        find = True
    else:
        packet.append(0) # Add highest priority
    return find, packet

def hash(key, width, depth):
    h = (key+(depth+1)**(depth)) % width
    return h

def sketchAction(switch, f_id, table, add_value, clear=False):
    global packet_collision, flow_collision, pkt_collision_counter, flow_collision_counter
    get_value = []
    for i in range(SKETCH_DEPTH):
        key = hash(f_id, len(table[i]), i)
        table[i][key] += add_value
        get_value.append(table[i][key])
        if clear:
            table[i][key] = 0
        # ------ Record ------
        if table == switch.packet_count_table: # Add packet count
            if add_value != 0:
                if f_id not in switch.packet_collision[i][key]:
                    if switch.packet_collision[i][key] != []:
                        switch.pkt_collision_counter += 1
                        print("Packet Count Collision - table ", i, ": ", f_id, " and ", switch.packet_collision[i][key])
                    switch.packet_collision[i][key].append(f_id)
                    print("put fid into packet collision table[",i,"][",key,"]")
            if clear == True:
                # print(packet_collision[i][key])
                if f_id in switch.packet_collision[i][key]:
                    switch.packet_collision[i][key].remove(f_id)
                # print("Clear key in packte size: ", f_id)
        elif table == switch.flow_size_table: # Add flow size
            if add_value != 0:
                if f_id not in switch.flow_collision[i][key]:
                    if switch.flow_collision[i][key] != []:
                        switch.flow_collision_counter += 1
                        print("Flow Size Collision - table ", i, ": ", f_id, " and ", switch.flow_collision[i][key])
                    switch.flow_collision[i][key].append(f_id) 
                    print("put fid into flow collision table[",i,"][",key,"]")
            if clear == True:
                # print(flow_collision[i][key])
                if f_id in switch.flow_collision[i][key]:
                    switch.flow_collision[i][key].remove(f_id)
                # print("Clear key in flow size: ", f_id)
        # ------ Record ------
    return min(get_value)

def updatePacketCntTable(switch, f_id, packet):
    cnt = sketchAction(switch, f_id, switch.packet_count_table, 1, False)
    if cnt == 1 or cnt == PACKET_CNT_THRESHOLD: 
        return True
    else:
        return False
    
def updateFlowSizeTable(switch, f_id, packet):
    global sketch_flow_size
    size = sketchAction(switch, f_id, switch.flow_size_table, packet[6], False)
    # Record
    if f_id not in switch.sketch_flow_size.keys():
        switch.sketch_flow_size[f_id] = []
    if len(switch.sketch_flow_size[f_id]) < PACKET_CNT_THRESHOLD:
        switch.sketch_flow_size[f_id].append(packet[6])
    # Record
    return size

def normalize(f_id2, packet_m, arrival_t):
    feature_time = abs(arrival_t - switch.flow_record_table[f_id2][4]) / (max_data[0]-min_data[0])
    normalize_packet1 = (packet_m - min_data[1]) / (max_data[1] - min_data[1])
    normalize_packet2 = (switch.flow_record_table[f_id2][5] - min_data[1]) / (max_data[1] - min_data[1])
    return np.array([[feature_time, normalize_packet1, normalize_packet2]])

# function optimized to run on gpu 
#@jit(target_backend='cuda') 
def classify(switch, f_id, packet, packet_m, arrival_t):
    if len(switch.coflow_queue.keys()) == 0: # Create a new queue
        return packet[0] # Real coflow ID
    sameScore = []
    diffScore = []
    sorted_coflow_keys = sorted(switch.coflow_queue.keys())
    for i in range(len(sorted_coflow_keys)):
        sameScore.append(0)
        diffScore.append(0)
        cnt = 0
        sampleNum = min(len(switch.coflow_queue[sorted_coflow_keys[i]][0]), 20) #min(num of flows in queue, 20)
        sampleList = random.sample(range(len(switch.coflow_queue[sorted_coflow_keys[i]][0])), sampleNum)
        for j in sampleList: # Each flow in coflow
            if switch.coflow_queue[sorted_coflow_keys[i]][0][j] not in switch.flow_record_table.keys():
                continue
            n = normalize(switch.coflow_queue[sorted_coflow_keys[i]][0][j], packet_m, arrival_t)
            predict_prob = MODEL.predict(n)
            predict_classes = predict_prob[0]
            sameScore[i] += predict_classes[1]
            diffScore[i] += predict_classes[0]
            cnt += 1
            # ------ Record ------
            switch.DNN_counter += 1
            if packet[0] == switch.coflow_queue[sorted_coflow_keys[i]][1][j] and predict_classes[1] > predict_classes[0]:
                switch.DNN_right += 1
            if packet[0] != switch.coflow_queue[sorted_coflow_keys[i]][1][j] and predict_classes[1] <= predict_classes[0]:
                switch.DNN_right += 1
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
        if len(switch.coflow_queue.keys()) < COFLOW_TABLE_SIZE:
            c_id = packet[0] # Real coflow ID
        else:
            # Find the smallest coflow
            small = [0, sys.maxsize] # [c_id, #]
            for i in range(len(sorted_coflow_keys)):
                if len(switch.coflow_queue[sorted_coflow_keys[i]][0]) < small[1]:
                    small[0] = sorted_coflow_keys[i]
                    small[1] = len(switch.coflow_queue[sorted_coflow_keys[i]][0])
            c_id = small[0] # Smallest coflow ID
    else:
        c_id = score[0] # Existing coflow ID
    return c_id

def groundtruth(switch, real_cid):
    result = {}
    for key in switch.coflow_queue.keys():
        result[key] = 0
    for key in switch.coflow_queue.keys():
        for cid in switch.coflow_queue[key][1]:
            if cid == real_cid:
                result[key]+=1
    return result

#label manually with some error probability
def label(packet,c_list,percentage):
    prob = random.randrange(0,100)
    if prob<percentage:
        return packet[0]
    else:
        cid = random.choice(c_list)
        while True:
            if cid != packet[0]:
                break
            else:
                cid = random.choice(c_list)
        return cid

def updateFlowRecordTable(switches, switch, f_id, c_list, packet):
    #flow_record_table
    #                                 0          1        2    3         4         5       6
    #(in Controller) Flow_ID(key), Coflow_ID, Priority, Size, TTL, Arrival_Time, Size_m, Finish
    # Get data from Packet Table
    cnt = sketchAction(switch, f_id, switch.packet_count_table, 0, False)
    size = sketchAction(switch, f_id, switch.flow_size_table, 0, False)
    if cnt == 1:
        print("Put ", f_id, "in Flow Table")
        if f_id in switch.flow_record_table.keys():
            print("(cnt = 1) Flow ", f_id, " is in Flow Table")
            switch.flow_record_table[f_id][6] = False #finish = false
            switch.flow_record_table[f_id][3] = INITIAL_TTL
        else:
            switch.flow_record_table[f_id] = [None, 0, size, INITIAL_TTL, packet[5], 0, False]
        return
    elif cnt == PACKET_CNT_THRESHOLD:
        sketchAction(switch, f_id, switch.packet_count_table, 0, True) # Reset
        # Classify
        packet_m = size / cnt
        # ------ Record ------
        if f_id in switch.sketch_flow_size.keys(): 
            real_packet_s = sum(switch.sketch_flow_size[f_id])
            real_packet_c = len(switch.sketch_flow_size[f_id])
            real_packet_m = real_packet_s/real_packet_c
            if math.isnan(abs(real_packet_s - size) / real_packet_s) == False and math.isnan(abs(real_packet_c - cnt) / real_packet_c) == False and math.isnan(abs(real_packet_m - packet_m) / (real_packet_s/real_packet_c)) == False:
                switch.sketch_size_err += abs(real_packet_s - size) / real_packet_s
                switch.sketch_cnt_err += abs(real_packet_c - cnt) / real_packet_c
                switch.sketch_mean_err += abs(real_packet_m - packet_m) / (real_packet_s/real_packet_c)
                switch.sketch_counter += 1
                print("------ ", f_id, " ------")
                print("sketch size: ", size, " real size: ", real_packet_s)
                print("sketch cnt: ", cnt, " real cnt: ", real_packet_c)
                print("sketch mean: ", packet_m, " real mean: ", real_packet_s/real_packet_c)
                print("-----------------")
        # ------ Record ------
        if f_id not in switch.flow_record_table.keys():
            print("(cnt = ", PACKET_CNT_THRESHOLD, ") Flow ", f_id, " is not in Flow Table")
            switch.flow_record_table[f_id] = [None, 0, size, INITIAL_TTL, packet[5], packet_m, False]
        else:
            switch.flow_record_table[f_id][2] = size
            switch.flow_record_table[f_id][5] = packet_m
        arrival_t = switch.flow_record_table[f_id][4]
        c_id = classify(switch, f_id, packet, packet_m, arrival_t)
        #inter_classify_cid = classify(switches[1-switches.index(switch)], f_id, packet, packet_m, arrival_t)

        #label manually with some error probability
        #c_id = label(packet, c_list, 80)

        print(c_id)
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
        if c_id in switch.coflow_queue.keys(): # Record in Coflow Queue for classify
            switch.coflow_queue[c_id][0].append(f_id)
            switch.coflow_queue[c_id][1].append(packet[0]) # Real coflow id of this flow 
        else:
            switch.coflow_queue[c_id] = [[f_id],[packet[0]]] #new queue
        if c_id in switch.coflow_priority_table.keys(): # Update priority
            priority = switch.coflow_priority_table[c_id][1]
        else: # New coflow
            switch.coflow_priority_table[c_id] = [0, 0] #size, priority
        # Update Flow Table
        switch.flow_record_table[f_id][0] = c_id
        switch.flow_record_table[f_id][1] = priority
        # Insert to Priority Table
        if len(switch.priority_table) < PRIORITY_TABLE_SIZE:
            switch.priority_table[f_id] = priority
        else:
            print("(Priority Table) Overflow")
            # Todo
        
        '''
        #calculate major and minor classify accuracy via groundtruth
        major_groundtruth = groundtruth(switch, packet[0]) #input real coflow id to find groundtruth
        minor_groundtruth = groundtruth(switches[1-switches.index(switch)], packet[0])
        print("major_groundtruth: ", major_groundtruth)
        print("minor_groundtruth: ", minor_groundtruth)
        print("major accuracy =", major_groundtruth[c_id]/sum(major_groundtruth.values())*100, "%")
        print("minor accuracy =", minor_groundtruth[inter_classify_cid]/sum(minor_groundtruth.values())*100, "%")
        '''

def schedule(table):
    for c_id in table.keys():
        size = table[c_id][0] * 1024 ##
        curPriority = 0
        tmp = INIT_QUEUE_LIMIT
        while size > tmp:
            curPriority += 1
            tmp *= JOB_SIZE_MULT
            if curPriority >= NUM_JOB_QUEUES:
                break
        table[c_id][1] = curPriority
    return table

def controllerUpdate(switch):
    # Update Flow Table (Size)
    coflow_size = {} # Coflow id, coflow size 
    for f_id in switch.flow_record_table.keys():
        if switch.flow_record_table[f_id][6] == False: #if flow not finished
            size = sketchAction(switch, f_id, switch.flow_size_table, 0, False)
            switch.flow_record_table[f_id][2] = size #update flow size
        # For next step
        if switch.flow_record_table[f_id][0] != None: #if flow has coflow id
            if switch.flow_record_table[f_id][0] not in coflow_size.keys():
                coflow_size[switch.flow_record_table[f_id][0]] = switch.flow_record_table[f_id][2]
            else:
                coflow_size[switch.flow_record_table[f_id][0]] += switch.flow_record_table[f_id][2]  
    # Update coflow size
    for c_id in switch.coflow_priority_table.keys():
        if c_id not in coflow_size.keys(): # Bug
            continue
        switch.coflow_priority_table[c_id][0] = coflow_size[c_id] 
    # Schedule
    switch.coflow_priority_table = schedule(switch.coflow_priority_table) # Update coflow priority
    # print("Coflow Table", coflow_priority_table)
    # Update Flow Table (Priority)
    update_flow_list = [] # Flow ID, Priority
    for f_id in switch.flow_record_table.keys():
        if switch.flow_record_table[f_id][0] != None: # Classified
            if switch.flow_record_table[f_id][1] != switch.coflow_priority_table[switch.flow_record_table[f_id][0]][1]: # Update priority
                switch.flow_record_table[f_id][1] = switch.coflow_priority_table[switch.flow_record_table[f_id][0]][1]
                update_flow_list.append([f_id, switch.flow_record_table[f_id][1]])
    # Update Priority Table
    for entry in update_flow_list:
        if entry[0] not in switch.priority_table.keys():
            print("(Update priority in Priority Table) Flow ", f_id, " is not in Priority Table")
            if len(switch.priority_table) < PRIORITY_TABLE_SIZE:
                switch.priority_table[f_id] = entry[1]
            else:
                print("(Priority Table) Overflow")
                # Todo
        else:
            switch.priority_table[entry[0]] = entry[1]
    return switch.coflow_priority_table

def controllerUpdateTTL(switch, f_id):
    clear_now = []
    finished_coflow = {}
    # Update TTL
    for f in switch.flow_record_table.keys(): 
        if f == f_id:
            switch.flow_record_table[f_id][3] = INITIAL_TTL
            switch.flow_record_table[f][6] = False
        else:
            switch.flow_record_table[f][3] -= 1
            if switch.flow_record_table[f][3] <= 0 and switch.flow_record_table[f][6] == False:
                print(counter, " ############### Clear", f_id)
                if switch.flow_record_table[f][0] == None: 
                    sketchAction(switch, f, switch.packet_count_table, 0, True)
                    clear_now.append(f)
                else: # Classified
                    if f in switch.priority_table.keys():
                        del switch.priority_table[f]
                        sketchAction(switch, f, switch.flow_size_table, 0, True)
                    switch.flow_record_table[f][6] = True
        if switch.flow_record_table[f][0] != None:
            if switch.flow_record_table[f][0] not in finished_coflow.keys():
                finished_coflow[switch.flow_record_table[f][0]] = True
            if switch.flow_record_table[f][6] == False: # Flow unfinished
                finished_coflow[switch.flow_record_table[f][0]] = False # Coflow unfinished
    # Delete finished coflows
    for c_id in finished_coflow.keys(): 
        if finished_coflow[c_id] == True:
            del switch.coflow_priority_table[c_id]
            for f in set(switch.coflow_queue[c_id][0]):
                if f in switch.flow_record_table.keys():
                    del switch.flow_record_table[f]
            del switch.coflow_queue[c_id]
    # Delete finished flows       
    for f in clear_now:
        del switch.flow_record_table[f]
    return

def PIFO(packet, wait_queue):
    #print("PIFO -> packet = ", packet)
    #print("PIFO -> packet type : ", type(packet))
    wait_queue.put((packet[-1],packet))
    #print("PIFO -> after put : ", wait_queue.get())
    return wait_queue

def egress(switch):
    item = switch.wait_queue.get()
    out_packet = item[1]
    switch.output_queue.append(out_packet)
    # ------ Record ------
    if out_packet[0] not in switch.coflow_completion.keys():
        switch.coflow_completion[out_packet[0]] = [counter, counter, 0, fb_coflow_size[str(out_packet[0])], fb_coflow_priority[str(out_packet[0])]]
    else:
        switch.coflow_completion[out_packet[0]][1] = counter
        switch.coflow_completion[out_packet[0]][2] = counter - switch.coflow_completion[out_packet[0]][0]
    # ------ Record ------
    return switch.output_queue

if __name__ == "__main__":
    readDataSet()
    print("Read packets data: ")
    input_queue, input_data_flow, f_id_list, c_list = loadCsvData()
    print(len(c_list), " coflows, ", len(f_id_list), " flows and ", len(input_queue), " packets")

    #sampling
    sample_limit = 100000
    sample_input_queue, sample_input_data_flow, sample_f_id_list, sample_c_list = sampling(input_queue, input_data_flow, f_id_list, c_list, 10)
    while len(sample_input_queue)>sample_limit:
        sample_input_queue, sample_input_data_flow, sample_f_id_list, sample_c_list = sampling(input_queue, input_data_flow, f_id_list, c_list, 10)
    print("After sampling: ")
    print(len(sample_c_list), " coflows, ", len(sample_f_id_list), " flows and ", len(sample_input_queue), " packets")

    #grouping
    switches=[]
    numOfSwitches = 2
    for i in range(numOfSwitches):
        switches.append(Switch())
    switches = grouping(switches, sample_input_queue, sample_input_data_flow, sample_f_id_list, numOfSwitches)
    #switches, shuffle_cid_list = grouping2(switches, sample_input_queue, sample_input_data_flow, sample_f_id_list, sample_c_list, numOfSwitches)
    for switch in switches:
        print(len(switch.input_queue))

    packet_index=-1
    while True:
        counter+=1
        packet_index+=1
        done=0
        for switch in switches:
            if counter >= len(switch.input_queue) and switch.wait_queue.qsize() == 0:
                done+=1
                continue
            if packet_index < len(switch.input_queue):
                this_packet = list(switch.input_queue[packet_index])
                f_id = getFlowID(this_packet, sample_f_id_list)
                find, this_packet = checkPriorityTable(switch, f_id, this_packet)
                if not find:
                    # Update Packet Count Table
                    action = updatePacketCntTable(switch, f_id, this_packet)
                # Update Flow Size Table
                updateFlowSizeTable(switch, f_id, this_packet)
                # New flow or Packet full, inform controller
                if not find and action:
                    updateFlowRecordTable(switches, switch, f_id, sample_c_list, this_packet)
                #print("flow record table: ", switch.flow_record_table)
                # Controller update
                if counter % CONTROLLER_UPDATE_TIME == 0 or packet_index == len(switch.input_queue)-1:
                    switch.coflow_priority_table = controllerUpdate(switch)
                #PIFO
                switch.wait_queue = PIFO(this_packet, switch.wait_queue)
            # Egress
            if counter % EGRESS_RATE == 0:
                switch.output_queue = egress(switch)
            # Print Result
            if counter % 100 == 0:
                print("Switch ", switches.index(switch))
                print("Time slot: ", counter)
                print("Size of Priority Table: ", len(switch.priority_table.keys()))
                switch.priority_table_time.append(counter)
                switch.priority_table_size.append(len(switch.priority_table.keys()))
                if switch.DNN_counter != 0:
                    print("DNN Accuracy: ", switch.DNN_right / switch.DNN_counter * 100, " %")
                if switch.sketch_counter != 0:
                    print("Sketch Count Err: ", switch.sketch_cnt_err / switch.sketch_counter * 100, " %")
                    print("Sketch Size Err: ", switch.sketch_size_err / switch.sketch_counter * 100, " %")
                    print("Sketch Mean Err: ", switch.sketch_mean_err / switch.sketch_counter * 100, " %")
                print("len of wait queue: ", len(switch.wait_queue.queue))
            # Update TTL
            controllerUpdateTTL(switch, f_id)
        if done==numOfSwitches:
            break    
    print("All switches complete")
    # ------ Record ------
    for switch in switches:
        with open(OUTPUT_CSV+str(switches.index(switch)), "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["Time slot", "Size"])
            for i in range(len(switch.priority_table_time)):
                writer.writerow([switch.priority_table_time[i], switch.priority_table_size[i]])
            print("Write Completed")
        with open(OUTPUT_COMPLETION_TIME+str(switches.index(switch)), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Coflow ID", "Start Time", "Completion Time", "Duration Time", "Coflow Size", "Coflow Priority"])
            for k, v in switch.coflow_completion.items():
                tmp = [k]
                tmp.extend(v)
                writer.writerow(tmp)
    # ------ Record ------

