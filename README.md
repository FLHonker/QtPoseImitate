#　基于对抗网络的游戏人物动作模仿与实现

```
@author: Frank Liu
@time: 2019.1.18 - 2019.5.31
@project: 武汉理工大学本科毕业设计
```
使用Qt实现的基于openpose和pix2pix对抗网络的游戏人物动作模拟。本科毕业设计。

## Requirements

- Linux(Ubuntu18.04)
- CPU - for test
 NVIDIA GPU(12GB) + CUDA CuDNN7.4 - for train 
- Python >= 3.4
- PyTorch >= 1.0.0
- python-opencv >= 3.4.0
- Keras
- Qt 5.12

## Architecture

<pre><font color="#729FCF"><b>QtPoseImitate</b></font>
├── aboutdialog.cpp
├── aboutdialog.h
├── aboutdialog.ui
├── <font color="#8AE234"><b>getModels.sh</b></font>
├── <font color="#729FCF"><b>images</b></font>
│   ├── <font color="#8AE234"><b>mv_450x420.avi</b></font>
│   ├── <font color="#8AE234"><b>pbug3_450x420.avi</b></font>
│   ├── <font color="#8AE234"><b>pbug_man_450x420.avi</b></font>
│   └── pose_128px.ico
├── images.qrc
├── main.cpp
├── mainwindow.cpp
├── mainwindow.h
├── mainwindow.ui
├── <span style="background-color:#4E9A06"><font color="#3465A4">pytorch_pix2pix</font></span>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">checkpoints</font></span>
│   │   └── <span style="background-color:#4E9A06"><font color="#3465A4">pbug_pix2pix</font></span>
│   │       ├── <font color="#8AE234"><b>latest_net_D.pth</b></font>
│   │       ├── <font color="#8AE234"><b>latest_net_G.pth</b></font>
│   │       ├── model_version.txt
│   │       ├── <font color="#8AE234"><b>v3.0_120_net_D.pth</b></font>
│   │       └── <font color="#8AE234"><b>v3.0_120_net_G.pth</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">data</font></span>
│   │   ├── <font color="#8AE234"><b>aligned_dataset.py</b></font>
│   │   ├── <font color="#8AE234"><b>base_data_loader.py</b></font>
│   │   ├── <font color="#8AE234"><b>base_dataset.py</b></font>
│   │   ├── <font color="#8AE234"><b>image_dataset.py</b></font>
│   │   ├── <font color="#8AE234"><b>image_folder.py</b></font>
│   │   ├── <font color="#8AE234"><b>__init__.py</b></font>
│   │   └── <font color="#8AE234"><b>unaligned_dataset.py</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">datasets</font></span>
│   │   ├── <font color="#8AE234"><b>combine_A_and_B.py</b></font>
│   │   ├── <font color="#8AE234"><b>download_cyclegan_dataset.sh</b></font>
│   │   ├── <font color="#8AE234"><b>download_pix2pix_dataset.sh</b></font>
│   │   ├── <font color="#8AE234"><b>make_dataset_aligned.py</b></font>
│   │   └── <span style="background-color:#4E9A06"><font color="#3465A4">pbug_full</font></span>
│   │       ├── <span style="background-color:#4E9A06"><font color="#3465A4">test</font></span>
│   │       │   └── <font color="#AD7FA8"><b>curPose.jpg</b></font>
│   │       ├── <font color="#729FCF"><b>train</b></font>
│   │       └── <font color="#729FCF"><b>val</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">docs</font></span>
│   │   ├── <font color="#8AE234"><b>datasets.md</b></font>
│   │   ├── <font color="#8AE234"><b>qa.md</b></font>
│   │   └── <font color="#8AE234"><b>tips.md</b></font>
│   ├── <font color="#8AE234"><b>environment.yml</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">keras_openpose</font></span>
│   │   ├── <font color="#8AE234"><b>config</b></font>
│   │   ├── <font color="#8AE234"><b>config.py</b></font>
│   │   ├── <font color="#8AE234"><b>config_reader.py</b></font>
│   │   ├── <font color="#8AE234"><b>keras_openpose_test.py</b></font>
│   │   ├── <span style="background-color:#4E9A06"><font color="#3465A4">model</font></span>
│   │   │   ├── <font color="#8AE234"><b>get_keras_model.sh</b></font>
│   │   │   └── <span style="background-color:#4E9A06"><font color="#3465A4">keras</font></span>
│   │   │       └── <font color="#8AE234"><b>model.h5</b></font>
│   │   ├── <font color="#8AE234"><b>model.py</b></font>
│   ├── <font color="#8AE234"><b>LICENSE</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">models</font></span>
│   │   ├── <font color="#8AE234"><b>base_model.py</b></font>
│   │   ├── <font color="#8AE234"><b>cycle_gan_model.py</b></font>
│   │   ├── <font color="#8AE234"><b>__init__.py</b></font>
│   │   ├── <font color="#8AE234"><b>networks.py</b></font>
│   │   ├── <font color="#8AE234"><b>pix2pix_model.py</b></font>
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">options</font></span>
│   │   ├── <font color="#8AE234"><b>base_options.py</b></font>
│   │   ├── <font color="#8AE234"><b>__init__.py</b></font>
│   │   ├── <font color="#8AE234"><b>test_options.py</b></font>
│   │   └── <font color="#8AE234"><b>train_options.py</b></font>
│   ├── <font color="#729FCF"><b>pbug_pix2pix</b></font>
│   │   └── <font color="#729FCF"><b>test_latest</b></font>
│   │       └── <font color="#729FCF"><b>images</b></font>
│   │           ├── <font color="#AD7FA8"><b>curPose_fake_B.jpg</b></font>
│   │           ├── <font color="#AD7FA8"><b>curPose_real_A.jpg</b></font>
│   │           └── <font color="#AD7FA8"><b>curPose_real_B.jpg</b></font>
│   ├── pix2pix_class.py
│   ├── <font color="#8AE234"><b>pix2pix_test.py</b></font>
│   ├── pubgPoseFake.py
│   ├── <span style="background-color:#4E9A06"><font color="#3465A4">scripts</font></span>
│   │   ├── <font color="#8AE234"><b>conda_deps.sh</b></font>
│   │   ├── <font color="#8AE234"><b>download_cyclegan_model.sh</b></font>
│   │   ├── <font color="#8AE234"><b>download_pix2pix_model.sh</b></font>
│   │   ├── <span style="background-color:#4E9A06"><font color="#3465A4">edges</font></span>
│   │   │   ├── <font color="#8AE234"><b>batch_hed.py</b></font>
│   │   │   └── <font color="#8AE234"><b>PostprocessHED.m</b></font>
│   │   ├── <span style="background-color:#4E9A06"><font color="#3465A4">eval_cityscapes</font></span>
│   │   │   ├── <span style="background-color:#4E9A06"><font color="#3465A4">caffemodel</font></span>
│   │   │   │   └── <font color="#8AE234"><b>deploy.prototxt</b></font>
│   │   │   ├── <font color="#8AE234"><b>cityscapes.py</b></font>
│   │   │   ├── <font color="#8AE234"><b>download_fcn8s.sh</b></font>
│   │   │   ├── <font color="#8AE234"><b>evaluate.py</b></font>
│   │   │   └── <font color="#8AE234"><b>util.py</b></font>
│   │   ├── <font color="#8AE234"><b>install_deps.sh</b></font>
│   │   ├── <font color="#8AE234"><b>test_before_push.py</b></font>
│   │   ├── <font color="#8AE234"><b>test_cyclegan.sh</b></font>
│   │   ├── <font color="#8AE234"><b>test_pix2pix.sh</b></font>
│   │   ├── <font color="#8AE234"><b>test_single.sh</b></font>
│   │   ├── <font color="#8AE234"><b>train_cyclegan.sh</b></font>
│   │   └── <font color="#8AE234"><b>train_pix2pix.sh</b></font>
│   ├── <font color="#8AE234"><b>test.py</b></font>
│   ├── <font color="#8AE234"><b>train.py</b></font>
│   └── <span style="background-color:#4E9A06"><font color="#3465A4">util</font></span>
│       ├── <font color="#8AE234"><b>get_data.py</b></font>
│       ├── <font color="#8AE234"><b>html.py</b></font>
│       ├── <font color="#8AE234"><b>image_pool.py</b></font>
│       ├── <font color="#8AE234"><b>__init__.py</b></font>
│       └── <font color="#8AE234"><b>visualizer.py</b></font>
├── QtPoseImitate.pro
├── QtPoseImitate.pro.user
├── READMR.md
└── <font color="#729FCF"><b>result</b></font>
    ├── <font color="#AD7FA8"><b>fake_out.avi</b></font>
    └── <font color="#AD7FA8"><b>pose_out.avi</b></font>
</pre>

## CMU - OpenPose

使用预训练好的OpenPose COCO模型，使用[下载脚本][3]自动下载到`pytorch_pix2pix/keras_openpose/model/keras/`，大约209MB，在[`keras_openpose_test.py`][5]中提供API，请自行查看注释，支持视频和图片直接输入，计算输出结果图/视频。

## pytorch-pix2pix

reference _[pix2pix README][1]_

### pubg(刺激战场) dataset

pix2pix数据集图片格式参照_[pix2pix Dataset][2]_，请放置于[`pytorch_pix2pix/datasets/`][4]目录下。本项目官方数据集下载地址:baiduyun
链接:<https://pan.baidu.com/s/1z9LjKF7WuIpCUoMir0uFzA>  提取码:1d6n

## License

本项目遵循开源协议MIT，其中不乏使用了一些开源代码，但是禁止商用！仅供相关领域爱好者研究学习！最终解释权归**Frank**(<frankliu624@outlook.com>)所有。
如使用过程中存在疑问，而README中介绍不详细，欢迎联系Frank帮您解决。

## Citation

```
@article{frank2019pubg,
  title         = {基于对抗网络的游戏人物动作模仿与实现},
  author        = {Frank Liu},
  year          = {2019},
}
```


[1]:https://github.com/FLHonker/QtPoseImitate/blob/master/pytorch_pix2pix/README.md

[2]:https://github.com/FLHonker/QtPoseImitate/blob/master/pytorch_pix2pix/docs/datasets.md

[3]:pytorch_pix2pix/keras_openpose/model/get_keras_model.sh

[4]:pytorch_pix2pix/datasets/

[5]:pytorch_pix2pix/keras_openpose/keras_openpose_test.py

