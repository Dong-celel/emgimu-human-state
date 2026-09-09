# 数据与实验协议

## 标签定义

`Direction` 描述当前运动方向，不描述手臂位置。`NONE` 表示确实没有方向；
`UNKNOWN` 表示本应有答案但数据无效、处于未标注回程或无法可靠判断。

`Gesture` 描述当前手部姿态。`NEUTRAL` 是自然放松；`OPEN` 是主动伸展后的
张开姿态。张开后 EMG burst 下降不自动把姿态改回 Neutral，释放过程由
`P_hand=RELEASE` 表示。

## 正式采集

- 四个 session，至少跨两天；每次摘下并重新佩戴。
- 每个 session 采集全部 28 个 `Direction × Gesture` 组合，每个组合 3 次。
- 复合动作三次分别采用手势相对手臂运动 `-200/0/+200 ms` 的提示偏移。
- trial 顺序随机；去程是正式标签区，未标注回程设为 Unknown。
- 程序记录 cue 时间；稳定区间使用 `stable_mask=true`，边界、回程和操作者
  判为不合格的片段设为 false。
- 每个 session 前单独录制约 60 秒校准块：静息、舒适强度的 Pinch/Fist/Open、
  六方向中性手势移动。无需要求最大自主收缩。

数据固定分区：Session 1–2 train，Session 3 validation，Session 4 test。
Session 4 只允许在冻结全部算法后以 `--unlock-test` 打开一次。
首次评估会在数据根目录留下带模型哈希的 `SESSION4_FINAL_RESULT.json` 锁文件。

## Trial NPZ

| 字段 | 形状 | 含义 |
|---|---:|---|
| `timestamp_ms` | `[N]` | 严格递增的设备源时间戳 |
| `emg` | `[N,8]` | 原始同步 EMG |
| `accel` | `[N,3]` | 原始加速度 |
| `gyro` | `[N,3]` | 原始角速度 |
| `direction` | `[N]` 或标量 | Direction 数值 |
| `gesture` | `[N]` 或标量 | Gesture 数值 |
| `stable_mask` | `[N]` | 可用于稳定分类训练的采样 |
| `session_id` | 标量字符串 | `1`–`4` |
| `trial_id` | 标量字符串 | 全局唯一 trial ID |
| `session_date` | 标量字符串 | 本地日期 `YYYY-MM-DD`，正式审计必需 |

运行正式审计：

```powershell
emgimu validate-dataset DATASET_ROOT --formal
```

## 校准文件

`DATASET_ROOT/calibration/session_N.json` 保存滤波选择、逐通道稳健尺度、重力、
陀螺仪偏置、设备到人体坐标旋转、腕带循环偏移和运动尺度。Session 4 校准
不更新分类器权重。

## 指标

最终报告同时给出 D/H macro-F1、联合准确率、Unknown 覆盖率、ECE、混淆矩阵、
端到端 onset-to-stable 延迟 P50/P95。Unknown 在 macro-F1 与联合准确率中按错误
计算。没有真实 Session 4 延迟标注时不得宣称满足 300 ms。
