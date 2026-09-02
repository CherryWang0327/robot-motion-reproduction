# GMR 与 ProtoMotions/PyRoki 对照实验简报

## 实验目的

使用相同的 `happy`、`taichi1` SMPL 动作，比较 GMR 与 ProtoMotions/PyRoki 生成的 G1 参考轨迹。两组参考轨迹最终均使用同一套 Whole Body Tracking（WBT）参数训练，以判断跟踪误差主要来自重映射方法，还是来自 RL/控制器。

## GMR 基线

两项 GMR 策略均完成整段确定性评测，无提前 termination。

| 动作 | 帧数/FPS | Anchor pos RMSE | Body pos RMSE | Body rot RMSE | Joint pos RMSE | Joint vel RMSE |
|---|---:|---:|---:|---:|---:|---:|
| happy | 885 / 50 | 0.0425 | 0.0259 | 0.0988 | 0.5230 | 4.0007 |
| taichi1 | 4003 / 50 | 0.2018 | 0.0291 | 0.0997 | 0.4361 | 1.0818 |

`happy` 的主要问题是关节位置和速度误差；`taichi1` 的主要异常是 anchor position 偏大。

## ProtoMotions/PyRoki 数据流程

流程参照 `/home/unitree/BysanRL/nogmr_motion_tracking.md`：

```text
SMPL CSV
→ ProtoMotions .motion
→ SMPL MotionLib .pt
→ PyRoki 重映射 G1 .pt
→ WBT 可训练 .npz
→ Whole Body Tracking RL
```

实际使用的脚本：

1. `all_convert_amass_to_proto.py`：CSV 转 `.motion`
2. `batch_pack_motion_to_pt.py`：`.motion` 打包为 SMPL PT
3. `retarget_amass_to_robot.sh`：PyRoki 重映射到 Unitree G1
4. `pt_to_npz_local.py`：G1 PT 转 WBT NPZ
5. `scripts/rsl_rl/train.py`：在本项目进行 Whole Body Tracking 训练

实际运行代码根为 `/home/unitree/projects/ProtoMotions`；`/home/unitree/BysanRL/nogmr_motion_tracking.md` 仅作为老师教程和参数规范。教程所需接口已适配到 projects 仓库。

两份最终 WBT 输入均已验证：

| 动作 | Proto NPZ 帧数 | FPS | 关节数 | 刚体数 | NaN/Inf | 与 GMR 对齐 |
|---|---:|---:|---:|---:|---|---|
| happy | 885 | 50 | 29 | 30 | 无 | 是 |
| taichi1 | 4003 | 50 | 29 | 30 | 无 | 是 |

原始 happy CSV 为 886 帧；按 50 FPS 转 WBT NPZ 后为 885 帧，恰好与 GMR happy 的 885 帧一致。taichi1 两侧均为 4003 帧。

## WBT 控制变量

GMR 与 Proto 两组统一采用：

- task：`Tracking-Flat-G1-v0`
- `num_envs=4096`
- `seed=0`
- 最少 30000 iterations
- 最多 100000 iterations
- checkpoint 间隔 500
- 收敛窗口 1000 iterations
- 相同平台期阈值
- 相同确定性评测、录像和 tracking error 导出

因此 RL 训练算法没有改变，主要自变量是 G1 参考动作的重映射来源。

## 教程参数复现

初次运行发现现有 PyRoki 环境只有 CPU JAX，导致教程的 1000 帧求解无法执行；该次小分块结果及其 WBT 训练已停止并移出正式实验目录。随后保持 JAX/JAXLIB 0.6.2 不变，补装同版本官方 CUDA 12 plugin/PJRT，并确认 `jax.devices()` 返回 RTX 5070 Ti 的 `CudaDevice(id=0)`。

正式重做使用教程原始脚本和参数：`chunk_len=1000`、`chunk_overlap=0`、`skip_freq=1`、`motion_filter=0`、输入/输出 50 FPS。正式 G1 PT、WBT NPZ 和 WBT 训练均从该重做结果重新生成。

## 当前状态与最终分析

- GMR happy、taichi1：训练和评测完成。
- Proto happy、taichi1：重映射和 WBT NPZ 转换完成。
- Proto happy：正在进行 WBT 训练。
- Proto taichi1：等待 happy 完成后自动训练。

最终比较完整执行、termination、收敛 iteration、六项 tracking RMSE、逐关节误差、脚部接触/滑动指标和视频自然度。若 Proto 显著改善对应异常，更支持重映射质量问题；若两种参考经过相同 WBT 训练后仍表现相近，则更支持 RL reward、控制带宽或机器人可达性限制。
