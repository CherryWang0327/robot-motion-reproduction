# GVHMR demo 的 SMPL CSV

本目录包含 10 个从 GVHMR `smpl_params_global` 提取并通过活动库筛选的项目 SMPL CSV，共 15,641 个 50 Hz 帧。

## 坐标与格式

- CSV 使用项目播放/部署约定：世界平移与根旋转转换为 **Z-up**，局部 body pose 保持原始 SMPL 关节基；平移单位为米，旋转为轴角弧度。
- 每行 80 列，与现有 LAFAN1 SMPL CSV 一致：`fps`、`global_orient[3]`、`body_pose[63]`、`transl[3]`、`betas[10]`。
- 30 Hz 源动作重采样到 50 Hz：根与 21 个身体关节采用 SO(3) SLERP，平移和 betas 采用线性插值。
- 内部 GMR 原始输入另存于 `/data/BysanRL/data_goal/csv/GVHMR/demo_gmr_raw`，该目录是 Y-up，不用于项目播放器。
- GMR 命令读取内部 raw CSV 且不传 `--z-trans`；项目 CSV 不直接送入 GMR，避免混用两套目标机器人轴约定。

## 文件

- `gvhmr_demo_smpl_csv_manifest.csv`：源 PT、匹配视频、FPS、帧数、坐标约定和 SHA-256。
- `gvhmr_demo_gmr_manifest.csv`：GMR 批处理输入清单。
- `gvhmr_demo_excluded_motions.csv`：已从活动库删除的 11 个短时或突变动作、判据和原始 PT 证据路径。
- `conversion_metadata.json`：转换器、列定义和重采样方法。

## 生成命令

```bash
/home/pcbysan/miniconda3/envs/bydmmc_nogmr/bin/python \
  /home/pcbysan/BysanRL/BeyondMimic_nogmr/experiments_nogmr/scripts/convert_gvhmr_pt_to_smpl_csv.py \
  --overwrite

/home/pcbysan/miniconda3/envs/bydmmc_nogmr/bin/python \
  /home/pcbysan/BysanRL/scripts/play_smpl_zup_mujoco_only.py \
  --mjcf /data/BysanRL/data_goal/xml/smpl_humanoid_zup.xml \
  --smpl_csv /data/BysanRL/data_goal/csv/GVHMR/demo/1438/1438_smpl_50fps.csv \
  --speed 1
```
