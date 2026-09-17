# 1. GVHMR

## 目录

- [[#1.1 将视频上传至 Ubuntu]]
- [[#1.2 运行GVHMR 视频动作重建]]
- [[#1.3 检查 GVHMR 输出]]
- [[#1.4 将 GVHMR PT 转为标准 SMPL CSV（50 FPS）]]

[[0. G1 人形机器人动作复现总线路]] ｜ [[2. 重定向]]

本页以 `march_video`（原地踏步）为例，记录视频动作重建与上游数据导出的独立操作内容。

本节将使用 **GVHMR** 对原地踏步视频进行处理，从普通 RGB 视频中提取人体运动信息，并重建对应的 **3D 人体姿态与运动序列**。

## 1.1 将视频上传至 Ubuntu

在本地电脑的终端中执行以下命令，将视频上传到 Ubuntu 设备：

```
scp <video_location_on_windows> unitree@192.168.31.149:/home/unitree/projects/GVHMR/docs/<folder_name>/
```

例如，将 `march_video.mp4` 上传到 `march_video` 文件夹：

```
scp C:\Users\username\Desktop\march_video.mp4 unitree@192.168.31.149:/home/unitree/projects/GVHMR/docs/march_video/
```

上传完成后，视频路径应为：

```
/home/unitree/projects/GVHMR/docs/march_video/march_video.mp4
```

## 1.2 运行GVHMR 视频动作重建

首先进入 GVHMR 项目目录，并启用已配置好的 Conda 环境：

```
cd /home/unitree/projects/GVHMR
conda activate gvhmr
```

环境激活完成后，执行 GVHMR 的 Demo 程序：

```
python tools/demo/demo.py \
  --video /home/unitree/projects/GVHMR/docs/march_video/march_video.mp4 \
  --output_root outputs \
  -s
```

参数说明：

- `--video`：指定待处理的视频文件。
    
- `--output_root outputs`：指定结果输出到项目目录下的 `outputs` 文件夹。
    
- `-s`：启用结果保存选项。
    

运行期间，GVHMR 会完成视频人物检测、人体运动估计以及三维 SMPL 人体运动重建。处理时间取决于视频时长和设备性能。

## 1.3 检查 GVHMR 输出

处理完成后，结果会保存在：

```
/home/unitree/projects/GVHMR/outputs/march_video/
```

本步骤的关键输出包括：

- `hmr4d_results.pt`：人体运动重建的原始结果数据。
    
- `1_incam.mp4`：相机坐标系下的 SMPL 人体重建视频。
    
- `2_global.mp4`：全局坐标系下的 SMPL 人体重建视频。

### 1.3.1 相机坐标系下的 SMPL 人体运动重建结果

文件路径：

```
/home/unitree/projects/GVHMR/outputs/march_video/1_incam.mp4
```

`1_incam.mp4` 展示的是**相机坐标系（in-camera）下的 SMPL 人体运动重建结果**。

该结果主要用于观察重建得到的三维人体模型与原始视频中人物之间的对应关系，例如人体姿态、腿部动作以及身体运动是否被正确识别。

可以使用以下命令打开视频：

```
xdg-open /home/unitree/projects/GVHMR/outputs/march_video/1_incam.mp4
```

打开后，检查 SMPL 人体模型是否能够正确跟随原视频中的人物完成原地踏步动作。

### 1.3.2 全局坐标系下的 SMPL 人体运动重建结果

文件：

```
/home/unitree/projects/GVHMR/outputs/march_video/2_global.mp4
```

`2_global.mp4` 展示的是**全局坐标系（global）下的 SMPL 人体运动重建结果**。

与 `1_incam.mp4` 不同，该视频更侧重于展示重建人体在三维空间中的整体姿态和运动，可以更加直观地观察人体的全局运动状态。

使用以下命令打开：

```
xdg-open /home/unitree/projects/GVHMR/outputs/march_video/2_global.mp4
```

确认两个视频中的人体动作合理后，即可使用 `hmr4d_results.pt` 作为后续动作处理与机器人动作生成流程的上游输入。

## 1.4 将 GVHMR PT 转为标准 SMPL CSV（50 FPS） [[GMR-(1)-PT--CSV]]

本步骤将 GVHMR 输出的 `hmr4d_results.pt` 转换为标准 SMPL CSV 格式。该 CSV 是 GMR 与当前 ProtoMotions / PyRoki 链路共用的上游输入：GMR 直接读取它；ProtoMotions 则将它传给 `all_convert_amass_to_proto.py`。

输入文件：

```
/home/unitree/projects/GVHMR/outputs/march_video/hmr4d_results.pt
```

输出文件：

```
/home/unitree/projects/GVHMR/outputs/march_video/march_video_smpl.csv
```

执行以下命令：

```
cd /home/unitree/projects/GVHMR

conda run -n gvhmr python scripts/gvhmr_to_smpl_csv.py \
  --input outputs/march_video/hmr4d_results.pt \
  --output outputs/march_video/march_video_smpl.csv \
  --fps 50
```

参数说明：

- `--input`：指定 GVHMR 生成人体运动重建结果 `hmr4d_results.pt`。
- `--output`：指定转换后的标准 SMPL CSV 文件保存位置。
- `--fps 50`：导出当前管线使用的 50 FPS SMPL CSV。不要将该共享源 CSV 误写为 30 FPS。

转换完成后，检查以下文件是否生成：

```
/home/unitree/projects/GVHMR/outputs/march_video/march_video_smpl.csv
```

该文件保存逐帧人体 SMPL 动作数据，是 GMR 的直接输入，也是 ProtoMotions 当前 CSV 输入分支的源文件。

