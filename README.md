# EMG–IMU Human State

这是一个独立于采集前端的实时分类后端。它接收共享源时间轴上的
`8 路 EMG + 3 路加速度 + 3 路角速度`（默认 200 Hz），每 40 ms
输出一次方向、手部姿态、两条动作阶段、激活强度、运动强度、一致性、
分项置信度和信号质量。

系统把方向和手势作为两个标签，而不是把 `Left+Fist` 扩成一个组合类：

```text
Direction = Unknown / None / Forward / Backward / Left / Right / Up / Down
Hand      = Unknown / Neutral / Pinch / Fist / Open
```

## 数据约定

每个 `.npz` trial 至少包含：

```text
timestamp_ms   [samples]
emg            [samples, 8]
accel          [samples, 3]
gyro           [samples, 3]
direction      [samples] 或标量
gesture        [samples] 或标量
session_id     标量字符串
trial_id       标量字符串
```

方向表示当前运动方向，不表示手臂所在位置。动作回程若未单独标注，必须设为
`Direction.UNKNOWN`，不能沿用去程标签。`Neutral` 是自然放松；`Open` 是主动
伸展后保持的张开姿态。

固定数据划分为 Session 1–2 训练、Session 3 验证、Session 4 最终测试。
工具会拒绝 trial 同时出现在多个分区中。

## 快速使用

```powershell
pip install -e .[dev]
emgimu validate-dataset path\to\dataset --formal
emgimu train-baseline path\to\dataset --output artifacts\baseline.pkl
emgimu evaluate path\to\dataset --model artifacts\baseline.pkl --split validation
emgimu train-neural path\to\dataset --output artifacts\dual_branch.pt
```

程序不会自动下载外部生理数据。`emgimu.external` 保存设备、许可证和允许的
适配策略；带 `NC` 的数据在商业模式下会被拒绝。

## 实时接口

`HumanStateEstimator.push_sample(...)` 接收一个源时间戳采样；累积到 200 ms
窗口后，以 25 Hz 返回 `HumanState`。新 OSC 地址为 `/emgimu/state/v2`。
旧 `/emgimu/state` 可由 `OscPublisher(publish_legacy=True)` 同时发送。

已有采集端只需向 `/emgimu/raw` 发送 `timestamp + 8 EMG + 3 accel + 3 gyro`：

```powershell
emgimu serve --model artifacts\baseline.pkl --calibration calibration\session_3.json `
  --publish-legacy --log-csv artifacts\runtime_observations.csv
```

观察数据足够后可拟合仅展示、不干预 D/H 的一致性模型：

```powershell
emgimu fit-consistency artifacts\runtime_observations.csv --output artifacts\consistency.json
```

最终验收不能用模拟数据替代：冻结模型后，只运行一次未查看过的 Session 4，
并把 Unknown 计作错误，检查 D/H 宏 F1、联合准确率和端到端 P95 延迟。
首次 `--unlock-test` 会写入 `SESSION4_FINAL_RESULT.json`；此后程序拒绝再次测试，
文件同时保存模型 SHA-256，避免测试后偷偷换模型。

在接入真实采集程序前，可用 `emgimu make-smoke-dataset TEMP_PATH` 生成明确标注为
非生理数据的管线测试集。完整字段、正式采集和协议说明见 `docs/`。

建议依次阅读：`docs/DATA_AND_EXPERIMENT.md`、`docs/OSC_V2.md`、
`docs/ARCHITECTURE_AND_LIMITS.md`。外部数据边界见 `docs/EXTERNAL_DATASETS.md`。
