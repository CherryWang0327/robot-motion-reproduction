# Real-Robot 160D Deployment Adapter

## My contribution

I authored the integration and safety adaptation that bridges an offline WBT robot-reference motion into a **160-dimensional real-robot policy observation**. The work was built by selectively reusing two existing reference implementations while keeping their responsibilities separate:

| Reference | What I retained or learned from it | What I changed or added in the 160D adapter |
|---|---|---|
| 177D SMPL deployment | ROS/DDS runtime structure, odometry handling, state machine, startup alignment, diagnostics | Replaced the SMPL command path with a robot-reference NPZ path; preserved the runtime structure while adding fail-closed gates |
| 154D robot-reference deployment | 58D reference convention, observation ordering, and robot-reference NPZ field semantics | Adapted only the observation contract; did not copy its command publishing, gains, action scaling, joint mapping, or startup behavior |

The resulting adapter is an independent development layer, not a copy of either teacher deployment.

## 160D observation contract I implemented

| Slice | Size | Content |
|---|---:|---|
| `0:58` | 58 | Reference robot joint position and velocity |
| `58:61` | 3 | Reference torso position relative to the current torso, expressed in the current torso frame |
| `61:67` | 6 | Relative torso orientation, represented by the first two rotation-matrix columns |
| `67:70` | 3 | Base linear velocity in base frame |
| `70:73` | 3 | Base angular velocity |
| `73:102` | 29 | Measured joint position minus nominal default |
| `102:131` | 29 | Measured joint velocity |
| `131:160` | 29 | Previous raw policy action |

This explicitly replaces the 75D custom SMPL command used by the 177D reference with a 58D robot-reference command sourced from WBT NPZ data.

## Engineering changes

- Implemented a robot-reference NPZ loader for `joint_pos`, `joint_vel`, `body_pos_w`, and `body_quat_w`.
- Reworked the main runtime loop to construct the 160D observation from reference motion, live robot state, odometry, IMU, and previous action.
- Replaced the SMPL CSV / coordinate-conversion input path with a robot-reference NPZ path and torso-based reference orientation.
- Preserved ROS odometry and safe runtime structure from the 177D reference while making the reference input and observation semantics explicit.
- Added offline validation that imports neither ROS nor Unitree DDS and cannot publish commands.
- Added dry-run behavior that does not construct a low-command publisher.
- Added stale-odometry, publisher-source, and explicit control-certification gates so an unreviewed run fails closed rather than publishing commands.
- Added diagnostics for odometry-versus-torso frame alignment, reference-vs-measured joint error, and observation dimension consistency.
- Added unit and contract tests for the 160D observation, reference adapter, frame handling, safety gates, and training interface.

## Verification boundary

The adapter has offline numerical and simulator-facing validation. Real-robot execution remains a separately reviewed operation: the physical source, latency, sign, and frame semantics of odometry must be verified with a no-command dry run and on-site safety approval. The public repository deliberately omits real-robot control code, deployment credentials, network configuration, and company-specific runtime artifacts.
