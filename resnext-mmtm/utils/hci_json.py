from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
import numpy as np

def convert_csv_to_dict(csv_path, subset, labels):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    key_start_frame = []
    key_end_frame = []

    duration_dic = {} # {label:[(ori_duration, key_duraion), ..], label:...}

    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        class_name = labels[row[1]-1]
        
        basename = str(row[0])
        start_frame = str(row[2])
        end_frame = str(row[3])

        keys.append(basename)
        key_labels.append(class_name)
        key_start_frame.append(start_frame)
        key_end_frame.append(end_frame)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        video_name = key.replace('./frames/', '').replace('/', '_')
        if key in database: # need this because I have the same folder 3  times
            key = key + '^' + str(i) 
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        start_frame = key_start_frame[i]
        end_frame = key_end_frame[i]

        # database[key]['annotations'] = {'label': label, 'start_frame':start_frame, 'end_frame':end_frame}

        ###### key frame
        key_frame_l = key_frames[video_name][str(start_frame)]

        # filter key frames < 4
        # if len(key_frame_l) < 4:
        #     del database[key]
        #     continue
        database[key]['annotations'] = {'label': label, 'start_frame':start_frame, 'end_frame':end_frame, 'key_frames':key_frame_l}
        if label in duration_dic:
            duration_dic[label].append((int(end_frame)-int(start_frame)+1, len(key_frame_l)))
        else:
            duration_dic[label] = [(int(end_frame)-int(start_frame)+1, len(key_frame_l))]

    return database, duration_dic

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(str(data.iloc[i, 1]))
    return labels

def convert_hci_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database, train_duration_dic = convert_csv_to_dict(train_csv_path, 'training', labels)
    val_database, val_duration_dic = convert_csv_to_dict(val_csv_path, 'validation', labels)
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    # with open(dst_json_path, 'w') as dst_file:
    #     json.dump(dst_data, dst_file)

    duration_dic = {}
    for label in train_duration_dic:
        duration_dic[label] = train_duration_dic[label]+val_duration_dic[label]

    analyze_duration('train', train_duration_dic)
    analyze_duration('val', val_duration_dic)
    analyze_duration('total', duration_dic)

def analyze_duration(subset, duration_dic):
    def print_statistics(ll, name):
        # print('%s mean %.3f, std %.3f, longest %d, shortest %d'%(name, np.mean(ll), np.std(ll), max(ll), min(ll)))
        print('%s mean %.2f, std %.2f, longest %d, shortest %d'%(name, np.mean(ll), np.std(ll), max(ll), min(ll)))

    print('############## analyze %s ##############'%subset)
    # with open('/data2/datasets/HCIGesture/duration_ori_key_%s.json'%subset, 'w') as f:
    #     json.dump(duration_dic, f)
    all_ori_duration, all_key_duration, all_none_key_duration = [], [], []
    for label in duration_dic:
        ori_duration = [t[0] for t in duration_dic[label]]
        key_duration = [t[1] for t in duration_dic[label] if t[1]>=3]
        none_key_duration = [t[0]-t[1] for t in duration_dic[label]]
        all_ori_duration += ori_duration
        all_key_duration += key_duration
        all_none_key_duration += none_key_duration
        print(label)
        print('key duration < 3', len(duration_dic[label])-len(key_duration), 'in %d samples'%len(duration_dic[label]))
        print_statistics(ori_duration, 'ori')
        print_statistics(key_duration, 'key')
        print_statistics(none_key_duration, 'none key')

    if subset == 'total':
        print_statistics(all_ori_duration, 'all ori')
        print_statistics(all_key_duration, 'all_key')
        print_statistics(all_none_key_duration, 'all_none key')

# def print_table():
#     with open('/data2/datasets/HCIGesture/duration_ori_key_train.json', 'r') as f:
#         train_duration_dic = json.load(f)
#     with open('/data2/datasets/HCIGesture/duration_ori_key_val.json', 'r') as f:
#         val_duration_dic = json.load(f)
#     with open('/data2/datasets/HCIGesture/duration_ori_key_total.json', 'r') as f:
#         total_duration_dic = json.load(f)
#     total_num = [0, 0]
#     train_num = [0, 0]
#     test_num = [0, 0]
#     for label in train_duration_dic:

#         print(label, len(total_duration_dic[label]), len(train_duration_dic[label]), len(val_duration_dic[label]))


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    with open('/data2/datasets/HCIGesture/key_frames.json', 'r') as f:
        key_frames = json.load(f)

    for class_type in ['all_but_None']:#['all', 'all_but_None', 'binary']:

        if class_type == 'all':
            class_ind_file = 'classIndAll.txt'
        elif class_type == 'all_but_None':
            class_ind_file = 'classIndAllbutNone.txt'
        elif class_type == 'binary':
            class_ind_file = 'classIndBinary.txt'


        label_csv_path = os.path.join(csv_dir_path, class_ind_file)
        train_csv_path = os.path.join(csv_dir_path, 'trainlist'+ class_type + '.txt')
        val_csv_path = os.path.join(csv_dir_path, 'vallist'+ class_type + '.txt')
        dst_json_path = os.path.join(csv_dir_path, 'hci' + class_type + '_keyframes_filter.json')

        convert_hci_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, dst_json_path)
        print('Successfully wrote to json : ', dst_json_path)
    # HOW TO RUN:
    # python nv_json.py '../annotation_nvGesture'
