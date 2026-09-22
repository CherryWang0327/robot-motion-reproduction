# GVHMR demo 的 G1 机器人动作 NPZ

本目录包含 10 个通过活动库筛选、可供 BYDMMC/IsaacLab 训练或回放的 G1 动作。

## 字段

- `fps`：50 Hz。
- `joint_pos`、`joint_vel`：`(T,29)`，分别为关节角和关节角速度。
- `body_pos_w`、`body_lin_vel_w`、`body_ang_vel_w`：`(T,30,3)`，机器人刚体的世界系位置、线速度和角速度。
- `body_quat_w`：`(T,30,4)`，IsaacLab **wxyz** 四元数。
- 世界坐标为 **Z-up**；长度单位为米，角度单位为弧度。
- 与公开项目 SMPL CSV 联合播放时默认启用 `--smpl_world_alignment gvhmr_gmr_root0`；该模式做严格的 -90° yaw 世界变换，并对齐首帧骨盆/机器人根节点 XY，不强行对齐骨盆高度。

## 时间采样说明

现有 `pkl_to_npz_local.py` 使用半开时间区间 `[0,duration)`。受浮点边界影响，NPZ 与 GMR PKL 帧数相同或少 1 帧，持续时间最多相差一个 50 Hz 采样周期（0.02 s）。清单逐动作记录了实际帧数。

## 文件

- `gvhmr_demo_robot_npz_manifest.csv`：PT、CSV、PKL、NPZ 的完整路径、SHA-256、帧数、坐标和校验误差。
- 每个动作位于同名子目录，文件名为 `<动作名>_gmr_50fps.npz`。

## 生成与校验命令

```bash
/home/pcbysan/miniconda3/envs/bydmmc/bin/python \
  /home/pcbysan/BysanRL/BeyondMimic/whole_body_tracking/scripts/pkl_to_npz_local.py \
  --input_path /data/BysanRL/data_goal/pkl/GVHMR_gmr/demo \
  --output_path /data/BysanRL/data_goal/npz/GVHMR_gmr/demo \
  --output_fps 50 --suffix '' --headless --device cpu \
  --input_root_rot_order xyzw

/home/pcbysan/miniconda3/envs/bydmmc/bin/python \
  /home/pcbysan/BysanRL/BeyondMimic_nogmr/experiments_nogmr/scripts/validate_gvhmr_gmr_outputs.py

/home/pcbysan/miniconda3/envs/bydmmc_nogmr/bin/python \
  /home/pcbysan/BysanRL/BeyondMimic_nogmr/experiments_nogmr/scripts/audit_gvhmr_gmr_motion_quality.py
```
