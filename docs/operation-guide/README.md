# G1 人形机器人动作复现：运行教程

本目录收录项目交付用的完整运行教程，覆盖从输入视频到 G1 Whole-Body Tracking（WBT）训练、评估及部署准备的全链路。

## 阅读顺序

1. [全链路概览](01-pipeline-overview.md)
2. [GVHMR：从视频重建 SMPL 动作](02-gvhmr.md)
3. [重定向：GMR 或 ProtoMotions / PyRoki](03-retargeting.md)
4. [Whole-Body Tracking：转换、训练与评估](04-whole-body-tracking.md)
5. [部署：Dry Run 与 Live Run](05-deployment.md)

> **重要：** GMR 路线与 ProtoMotions / PyRoki 路线二选一；不要混用两条路线产生的动作文件、训练产物或 checkpoint。
