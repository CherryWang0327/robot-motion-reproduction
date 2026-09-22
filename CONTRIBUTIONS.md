# 实现贡献总览 / Implementation contribution map

本仓库的贡献不只限于 `programs/` 中的两个独立入口。动作复现的编排、格式转换、验证、训练/评估、部署适配以及模块级实现共同构成完整工作；下表按职责定位代码，而不把它们误分为“外部源码”或“仅示例程序”。

The contribution is not limited to the two standalone entry points under `programs/`. Workflow orchestration, representation conversion, validation, training/evaluation, deployment adaptation, and module-level implementations together form the work. The table organizes code by responsibility rather than presenting it as merely external source or examples.

| 贡献模块 / Contribution area | 主要位置 / Primary location | 内容 / Scope |
|---|---|---|
| 工作流编排与接口契约 / Workflow orchestration and contracts | `g1-motion-pipeline/g1pipe/` | 视频输入、阶段编排、路线选择、产物契约、任务状态与 CLI。 |
| GVHMR→GMR 转换 / GVHMR-to-GMR conversion | `g1-motion-pipeline/g1pipe/front.py`, `convert.py`, `source/GVHMR/scripts/`, `source/GMR/scripts/` | SMPL CSV 规范、坐标约定、GMR 输入输出转换。 |
| GMR 与 ProtoMotions 路线衔接 / GMR and ProtoMotions route integration | `g1-motion-pipeline/g1pipe/gmr.py`, `proto_preview.py`, `source/GMR/`, `source/ProtoMotions/` | G1 重定向、路线产物衔接与回放。 |
| 参考动作验证与可视化 / Reference validation and visualization | `g1-motion-pipeline/g1pipe/validate.py`, `render_reference.py`, `programs/playback_scripts/` | NPZ/CSV/SMPL 回放、坐标检查、格式与时序验证。 |
| WBT 训练与评估 / WBT training and evaluation | `g1-motion-pipeline/g1pipe/train.py`, `source/whole_body_tracking/scripts/`, `source/whole_body_tracking/source/` | 训练启动、基线/对照实验、评估和诊断。 |
| 160D 部署适配 / 160D deployment adaptation | `source/whole_body_tracking/deployment/`, `g1-motion-pipeline/reference-implementations/160D/`, `programs/deploy_real/` | robot-reference 观测、策略输入检查、dry-run 和部署文档。 |

## 目录说明 / Directory roles

- `g1-motion-pipeline/`：主实现层；包含编排、转换、验证、训练、Web 控制台与展示证据。
- `source/`：按 GVHMR、GMR、ProtoMotions 和 WBT 模块保存的实现、脚本、配置和实验记录；它们是贡献的一部分，不是未参与的上游镜像。
- `programs/`：仅放可单独调用的部署、回放和验证工具；它不定义贡献边界。
- `reproduction_assets/`：可审计的输入、输出、原始视频和清单。
- `docs/`：原始流程文档与说明索引。
