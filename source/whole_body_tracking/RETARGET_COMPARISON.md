# GMR vs ProtoMotions/PyRoki 对照实验

## 固定实验条件

- 动作：`happy`、`taichi1`
- 人体源：同一份 50 FPS SMPL CSV
- 机器人：Unitree G1
- ProtoMotions 输出：50 FPS
- PyRoki：`skip_freq=1`，关闭 motion filter，避免额外平滑改变比较条件
- PyRoki JAX 求解严格使用教程默认 `chunk_len=1000`、`chunk_overlap=0`。现有 `pyroki` 环境保持 JAX/JAXLIB 0.6.2，并安装同版本官方 CUDA 12 plugin/PJRT；必须确认 `jax.devices()` 返回 `CudaDevice(id=0)`，不能使用 CPU fallback。
- RL：`Tracking-Flat-G1-v0`、4096 environments、seed 0、最少 30000、最多 100000、同一平台期标准
- 评价：完整执行、确定性录像、六项 tracking error、逐关节误差和失败原因

`happy` 的 CSV 是 886 帧，而既有 GMR NPZ 是 885 帧。保留两份原始产物，不静默删帧；最终误差横向比较使用共同的前 885 帧，完整执行结果同时报告各自原始帧数。

## 目录

每个动作位于 `retarget_comparison/protomotions/<motion>/`：

1. `source/`：经 SHA256 固定的原始 CSV 副本
2. `01_smpl_motion/`：ProtoMotions `.motion`
3. `02_smpl_pt/`：打包后的 SMPL `.pt`
4. `03_g1_pyroki/`：PyRoki G1 `.pt`
5. `04_wbt_npz/`：WBT 训练输入
6. `logs/`：每阶段日志

ProtoMotions 代码根固定为 `/home/unitree/projects/ProtoMotions`。老师教程缺少的 CSV、PT打包、完整帧分块和 filter 开关接口已适配到该 projects 仓库；原 projects 文件备份在 `/home/unitree/projects/ProtoMotions/_pre_tutorial_adapter_20260812/`。

## 执行

当前十动作基线未达到 10/10 时，脚本会拒绝运行 GPU 流程。基线完成后：

```bash
cd /home/unitree/projects/whole_body_tracking

/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_protomotions_comparison_pipeline.py --dry-run

/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_protomotions_comparison_pipeline.py
```

中断后可从阶段恢复，例如：

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_protomotions_comparison_pipeline.py \
  --motions happy --phase retarget
```

不要使用 `--ignore-baseline-lock`，除非明确决定让两个任务争用 GPU。

## Whole Body Tracking 对照训练

两个 PyRoki NPZ 校验通过后，使用专用入口顺序训练。该入口复用十动作 baseline 的 WBT 调度逻辑，不使用 ProtoMotions 的训练框架：

```bash
cd /home/unitree/projects/whole_body_tracking

/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_retarget_comparison_training.py \
  --num_envs 4096 \
  --min_iterations 30000 \
  --max_iterations 100000 \
  --window 1000 \
  --poll_seconds 60
```

顺序为 `proto_happy`、`proto_taichi1`。每个动作在收敛后自动导出训练报告、确定性评测、视频和 tracking error。状态文件为：

```text
results/retarget_comparison/orchestrator_state.json
```
