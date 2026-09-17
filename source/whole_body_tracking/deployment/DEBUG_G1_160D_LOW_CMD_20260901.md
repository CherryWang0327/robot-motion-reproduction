# G1 160D 实机默认姿态软态问题排查

日期：2026-09-01  
脚本：`deploy_real_robotref_160.py`  
配置：`configs/g1_robotref_160_taichi1.yaml`

## 本次状态

用户已确认：应用本次修改后实机运行成功。除本文“已实施修改”列出的内容外，不再继续调整控制逻辑、刚度或默认关节角。

## 现象

将脚本的 `dry_run` 设为 `False` 后：

```text
Enter zero torque state.
Waiting for the start signal...
Moving to default pos.
```

遥控器 Start 已被识别，但机器人没有保持刚度或到达 `default_angles`。

## 已确认的 DDS 配置

YAML：

```yaml
msg_type: "hg"
lowcmd_topic: "rt/lowcmd"
lowstate_topic: "rt/lowstate"
```

只读 DDS 验证：

```text
C++ SDK 0.10.2 read-only test on eno1
LOWSTATE_OK mode_machine=5
samples=82
```

进一步只读读取 HG LowState：

```text
tick=4483323
mode_machine=5
mode_pr=0
motor_count=35
```

结论：`hg`、`rt/lowcmd`、`rt/lowstate`、PR 模式和当前 G1 DDS 通信是匹配的；不是 LowState 话题或消息类型完全错误。

## LowCmd 发送代码链路

当 `dry_run=False` 且 `use_ros_cmd=False` 时：

```text
move_to_default_pos()
  -> maybe_send_low_cmd()
  -> send_cmd()
  -> CRC().Crc(cmd)
  -> lowcmd_publisher_.Write(cmd)
```

关键位置：

- `Controller.__init__()`：创建 `ChannelPublisher(config.lowcmd_topic, LowCmdHG)`。
- `move_to_default_pos()`：原设计在 6 秒内、每个 `control_dt=0.02 s` 写入目标 q、kp、kd。
- `maybe_send_low_cmd()`：非 ROS 命令路径转到 DDS `send_cmd()`。
- `send_cmd()`：计算 CRC 后调用 DDS `Write()`。

历史日志已确认多个运行打印过：

```text
dry_run       = False
Moving to default pos.
```

但旧版本没有 `Write()` 调用计数，因此不能仅靠旧日志证明电机侧确实收到了每一帧命令。

## MotionSwitcher 遗留控制权

首次只读查询：

```text
BEFORE: {'form': '0', 'name': 'ai'}
```

这表示机载 `ai` 运动模式仍占有控制权。官方 G1 低层示例在创建 `rt/lowcmd` 发布器前会：

```python
status, result = msc.CheckMode()
while result['name']:
    msc.ReleaseMode()
    status, result = msc.CheckMode()
```

160D 脚本原先没有该仲裁流程，因此 `Write()` 可能被机载模式拒绝或覆盖。

本次已按明确授权释放当前遗留 `ai`，且立即复查：

```text
BEFORE 0 {'form': '0', 'name': 'ai'}
RELEASE (0, None)
AFTER  0 {'form': '0', 'name': ''}
```

当前 `MotionSwitcher` 已为空闲。`ai` 占用不是之后一次测试仍软的唯一原因。

## 确定的默认姿态保持缺陷

旧流程为：

```text
Start
  -> move_to_default_pos()：只发送约 6 秒 LowCmd
  -> wait_for_obsrun_ready()：最长等待 300 秒，但不发送任何 LowCmd
  -> default_pos_state()
```

G1 低层控制需要持续接收命令。6 秒后脚本进入 `/Odometry_2` 等待且停止发送 LowCmd，电机命令超时后会回到软态。

这不表示 YAML 中的 kp 在第 6 秒被改为零；而是控制端停止持续发送 q/kp/kd 命令。

## 已实施修改

### 1. 退出时清理 MotionSwitcher

脚本现在在收到 Start 并进入控制会话后，以 `try/finally` 清理：

```text
发送阻尼命令
-> CheckMode()
-> ReleaseMode()
-> 再次 CheckMode()
```

预期退出输出：

```text
[CLEANUP] Damping command sent.
[CLEANUP] MotionSwitcher before release: ...
[CLEANUP] ReleaseMode status=0
[CLEANUP] MotionSwitcher after release: ... {'name': ''}
```

不会自动重新选择或启动 `ai`。

### 2. 等待 OBSRun 时保持默认姿态

新增 `hold_default_pos()`：

```python
motor_cmd.q = default_angles[j]
motor_cmd.dq = 0
motor_cmd.kp = stiffness[j]
motor_cmd.kd = damping[j]
motor_cmd.tau = 0
maybe_send_low_cmd()
```

`wait_for_obsrun_ready()` 现在每 20 ms 调用一次该函数，直到收到 `/Odometry_2` 或超时。它使用原 YAML 的 `default_angles`、`stiffness`、`damping`，没有提高刚度或改关节目标。

运行时每 5 秒打印：

```text
[WAIT /Odometry_2] Holding default pose; 295 s remaining.
```

### 3. 修正 HG 消息速度字段名

HG DDS `MotorCmd_` 的实际字段为：

```text
mode, q, dq, tau, kp, kd, reserve
```

脚本中原先的 `.qd = 0` 已统一修正为 `.dq = 0`。q/kp/kd 本身原本字段正确；该修正保证期望速度零值会被正确序列化。

## YAML 与索引检查

对 `g1_robotref_160_taichi1.yaml` 的只读检查：

```text
leg + arm/waist 索引长度：29
stiffness 长度：29
damping 长度：29
default_angles 长度：29
索引：0..28，29 个唯一值
stiffness：14.2506 .. 99.0984，无 0
damping：0.9072 .. 6.3088，无 0
default_angles：-0.363 .. 0.669
```

腿部索引为 `0..11`，腰/手臂为 `12..28`，与 G1 29 自由度编号一致；没有证据表明默认姿态命令写错到非腿部电机。

## 下次验证步骤

1. 结束旧进程并重启脚本；已运行的 Python 进程不会热加载本次源码修改。
2. 启动前只读确认：

   ```text
   CheckMode() -> {'name': ''}
   ```

3. 按 Start 后确认出现：

   ```text
   Moving to default pos.
   Robot is standing. Start OBSRun now; holding default pose while waiting for /Odometry_2...
   [WAIT /Odometry_2] Holding default pose; ...
   ```

4. 在 OBSRun 未启动的等待阶段，机器人应继续接收原始默认姿态 q/kp/kd，而不因等待 `/Odometry_2` 自动变软。
5. 退出时确认：

   ```text
   [CLEANUP] MotionSwitcher after release: status=0, mode={'name': ''}
   ```

6. 如默认姿态阶段仍完全无响应，下一步仅增加发送诊断日志：每秒打印 `dry_run`、发布器是否存在、`Write()` 次数、`mode_machine`、以及电机 0/3 的 q/kp/kd。该日志可区分“未调用 Write”与“Write 后电机侧未接受”。

## 当前判断

已排除或弱化：

- `dry_run=True`：日志确认过 `False`。
- HG/GO 消息类型或 `rt/lowstate` 完全不匹配：LowState 已实际收到，且配置与官方 G1 示例一致。
- YAML 长度、kp/kd 全零、腿部电机索引错误：检查通过。

已确认需要处理：

- 退出后释放 MotionSwitcher 控制权，避免遗留 `ai`。
- 等待 OBSRun 时持续发送默认姿态，避免 LowCmd 超时后软态。
- 使用 HG 正确字段名 `dq`。

尚待通过发送计数日志确认：

- 默认姿态前 6 秒的每帧 DDS `Write()` 是否实际到达电机控制侧。
