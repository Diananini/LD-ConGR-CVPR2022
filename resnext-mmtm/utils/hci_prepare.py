import os
import numpy as np
import glob
import sys
import shutil
import pandas as pd

none_label = 11
dataset_path = "/data2/datasets/HCIGesture"
frames_dir = 'frames'
ext = '.png'
anno_path = 'annotation_hciGesture/dataV3'
def load_annos(file_with_split = './train_labels.txt'):
    label_file_path = os.path.join(dataset_path,file_with_split)
    data_dict = {}  # {video_name1: {label0:[(start, end), (start, end), ...], label1: [(start, end), (start, end), ...], ...},
                #  video_name2: {}, ... }

    with open(label_file_path) as label_file:
        for line in label_file:
            if line.strip():
                line_l = line.split(',')
                video_name = line_l[0]
                data_dict[video_name] = []

                for gesture in line_l[1:]:
                    start_f, end_f, label = gesture.split(' ')
                    data_dict[video_name].append((int(start_f), int(end_f), int(label)))

    return data_dict

def create_list(data_dict, sensor,  class_types = 'all'):
    new_lines = []
    for video_name in sorted(data_dict.keys()):
        folder_path = os.path.join('./'+frames_dir, video_name +'_'+ sensor + '_all')
        n_images = len(glob.glob(os.path.join(dataset_path, folder_path, '*%s'%ext)))
        start = 1

        for gesture in data_dict[video_name]:
            g_start, g_end, label = gesture

            if g_end-g_start+1 < 4: # filter the gestures short than 4 frames
                continue

            new_label = label + 1
            
            if class_types == 'all':
                if (g_start - start) >= 8:# Some action starts right away so I do not add None LABEL
                    new_lines.append(folder_path + ' ' + str(none_label)+ ' ' + str(start)+ ' ' + str(g_start-1))
                new_lines.append(folder_path + ' ' + str(new_label)+ ' ' + str(g_start)+ ' ' + str(g_end))
            elif class_types == 'all_but_None':
                new_lines.append(folder_path + ' ' + str(new_label)+ ' ' + str(g_start)+ ' ' + str(g_end))
            elif class_types == 'binary':
                if (g_start - start) >= 8:# Some action starts right away so I do not add None LABEL
                    new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(g_start-1))
                new_lines.append(folder_path + ' ' + '2' + ' ' + str(g_start)+ ' ' + str(g_end))
            start = g_end+1

        if (n_images - start >4):
            if class_types == 'all':
                new_lines.append(folder_path + ' ' + str(none_label) + ' ' + str(start)+ ' ' + str(n_images))
            elif class_types == 'binary':
                new_lines.append(folder_path + ' ' + '1' + ' ' + str(start)+ ' ' + str(n_images))

    return new_lines

    
if __name__ == "__main__":
    # sensors = ["color", "depth"]
    # extract_frames(sensors=sensors)
    
    # mkdir annotation_mouseGesture
    # python utils/mouse_prepare.py training trainlistall.txt all
    # python utils/mouse_prepare.py training trainlistall_but_None.txt all_but_None
    # python utils/mouse_prepare.py training trainlistbinary.txt binary
    # python utils/mouse_prepare.py validation vallistall.txt all
    # python utils/mouse_prepare.py validation vallistall_but_None.txt all_but_None
    # python utils/mouse_prepare.py validation vallistbinary.txt binary

    subset = sys.argv[1]
    file_name = sys.argv[2]
    class_types = sys.argv[3]

    sensors = ["color"]
    file_lists = dict()
    if subset == 'training':
        file_list = "./train_labels.txt"
    elif subset == 'validation':
        file_list = "./test_labels.txt"
    
    data_dict = load_annos(file_with_split = file_list)

    print("Processing List")
    new_lines = create_list(data_dict = data_dict, sensor = sensors[0], class_types = class_types)

    print("Writing to the file ...")
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    if not os.path.exists(os.path.join(anno_path, 'classIndAllbutNone.txt')):
        shutil.copy(os.path.join(dataset_path, 'classIndAllbutNone.txt'), anno_path)
    label2class = pd.read_table(os.path.join(dataset_path, 'classIndAllbutNone.txt'), sep=' ', header=None)
    all_classes = [class_name for class_name in label2class[1]]
    if not os.path.exists(os.path.join(anno_path, 'classIndAll.txt')):
        with open(os.path.join(anno_path, 'classIndAll.txt'), 'w') as f:
            for i, class_name in enumerate(all_classes):
                f.write('%d %s\n'%(i+1, class_name))
            f.write('%d none'%(i+2))
    if not os.path.exists(os.path.join(anno_path, 'classIndBinary.txt')):
        with open(os.path.join(anno_path, 'classIndBinary.txt'), 'w') as f:
            f.write('1 none\n')
            f.write('2 gesture')
    file_path = os.path.join(anno_path, file_name)
    with open(file_path, 'w') as myfile:
        for new_line in new_lines:
            myfile.write(new_line)
            myfile.write('\n')
    print("Scuccesfully wrote file to:",file_path)
    