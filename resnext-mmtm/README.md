## Get the dataset and models
Data access and download: refer to https://github.com/Diananini/LD-ConGR-CVPR2022#data-download

Data preprocessing: The annotations have been processed into the format required by the code and can be found in [annotation_hciGesture/dataV3](https://github.com/Diananini/LD-ConGR-CVPR2022/tree/main/resnext-mmtm/annotation_hciGesture/dataV3). All you need to do is extract frames of the with `ffmpeg`: `ffmpeg -i videoPath -r 30 framesPath/%05d.png`

The data is supposed to be structured as follows:
```
frames
├── 002
│  ├── g_s1_l1_00001_color_all
│  │   ├── 00001.png
│  │   ├── 00002.png
│  │   ├── ...
│  ├── g_s1_l1_00001_depth_all
│  │   ├── 00001.png
│  │   ├── 00002.png
│  │   ├── ...
│  ├── g_s1_l1_00002_color_all
│  ├── g_s1_l1_00002_depth_all
│  ├── g_s1_l1_00003_color_all
│  ├── g_s1_l1_00003_depth_all
│  ├── ...
├── 003
├── 004
├── ...
├── 030
├── 031
└── 032
```

Pretrained models on Jester dataset: [Google drive](https://drive.google.com/file/d/1JAYPxDNO5A9PvFdpRvVDzWNhYdKnhYz9/view?usp=sharing), [Baidu drive](https://pan.baidu.com/s/1BKPZRnFSXJMi7fAfre57jQ?pwd=41ym)

Models in the paper: [Google drive](https://drive.google.com/drive/folders/1yfjZhiRkBjQdlA3eMf9U0fMbzIcxlg5t?usp=sharing), [Baidu drive](https://pan.baidu.com/s/1wE7Ul5qgVpi6ROl_sQfLHA?pwd=x5ia)

## Run test
Run `bash sh/run_clf_hci_test.sh`

Arguments:

    `--root_path` The absolute path of `resnext-mmtm`
    `--video_path` The parent directory path of `frames`
    `--result_path` Results directory path
    `--resume_path` Model path
    `--sample_duration` Temporal duration of inputs
    `--model` Model name, 'resnext' or 'mmtnet'
    `--modality` Modality of input data, 'RGB' or 'RGB-D'
    `--iscrop` Use the gesture region estimation strategy
    `--isKeyframes` Use the key frame sampling strategy

## Run training
Run `bash sh/run_clf_hci_train.sh`

Arguments:

    `--root_path` The absolute path of `resnext-mmtm`
    `--video_path` The parent directory path of `frames`
    `--result_path` Results directory path
    `--resume_path` Model path
    `--sample_duration` Temporal duration of inputs
    `--model` Model name, 'resnext' or 'mmtnet'
    `--modality` Modality of input data, 'RGB' or 'RGB-D'
    `--iscrop` Use the gesture region estimation strategy
    `--resize` Resize the estimated gesture region
    `--resize_size` Resize size of the estimated gesture region
    `--isKeyframes` Use the key frame sampling strategy
    `--pretrain_path` Pretrained model path
    `--pretrain_dataset` Pretrained dataset