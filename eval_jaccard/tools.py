import os
import scipy.io as sio


def read_formatted_file(truth_file):
    truth = open(truth_file)
    gt_table = {}  # video_name: [((start,end), label), ...]
    js_table = {}  # video_name: 0
    for line in truth: # video_name s,e:label s,e:label
        if line[-1] == '\n':
            line = line[:-1]
        video_name, seg_line = line.split(' ', 1)
        segs = seg_line.split(' ')
        gt_table[video_name] = []
        js_table[video_name] = 0
        for seg in segs:
            seg, label = seg.split(':')
            s, e = seg.split(',')
            label, s, e = int(float(label)), int(s), int(e)
            gt_table[video_name].append(((s, e), label))
    return gt_table, js_table

def read_formatted_file_hci(truth_file):
    truth = open(truth_file)
    gt_table = {}  # video_name: [((start,end), label), ...]
    js_table = {}  # video_name: 0
    for line in truth: # video_name,s e label,s e label
        if line.strip():
            video_name, seg_line = line.strip().split(',', 1)
            segs = seg_line.split(',')
            gt_table[video_name] = []
            js_table[video_name] = 0
            for seg in segs:
                s, e, label = seg.split(' ')
                label, s, e = int(float(label)), int(s), int(e)
                gt_table[video_name].append(((s, e), label))
    truth.close()
    return gt_table, js_table

def formatted_write(table, out_f):
    out = open(out_f, 'w')
    for video in table:
        segments = table[video]
        out.write(video)
        for segs in segments:
            seg_points, label = segs
            s, e = seg_points
            out.write(' %d,%d:%d' % (s, e, label))
        out.write('\n')
    out.close()


def convert_format(mat_file, set_name, output_file_name):
    output = open(output_file_name, 'w')
    data = sio.loadmat(mat_file)

    x = data[set_name][0][0]
    video_name = x[1]
    all_rgb_names = video_name[0][0][0]
    # all_depth_names = video_name[0][0][1]
    all_labels = x[2]
    all_temp_segment = x[3]
    size = len(all_labels)

    for i in range(size):
        rgb_name = all_rgb_names[i][0][0]
        rgb_name = rgb_name.replace('\\', '/')
        # depth_name = all_depth_names[i][0][0]
        labels = all_labels[i][0][0]
        segments = all_temp_segment[i][0]
        print(rgb_name)
        output.write(rgb_name)
        for j in range(len(segments)):
            label = labels[j]
            seg = segments[j]
            print('\t%d at [%d to %d]' % (label, seg[0], seg[1]))
            output.write(' %d,%d:%d' % (seg[0], seg[1], label))
        output.write('\n')
    output.close()

if __name__ == '__main__':
    print("in Main")
    # convert_format('mats/test_label.mat', 'test', 'truth/test_truth.txt')
    # convert_format('mats/valid_label.mat', 'valid', 'truth/valid_truth.txt')
    #
