# OSC v2 协议

默认输出地址：

```text
/emgimu/state/v2
```

参数顺序固定为：

```text
timestamp_ms
direction_id
gesture_id
arm_phase_id
hand_phase_id
activation
motion_intensity
consistency_id
consistency_score
q_direction
q_gesture
q_arm_phase
q_hand_phase
emg_quality
imu_quality
quality_flags
```

枚举：

```text
Direction: -1 Unknown, 0 None, 1 Forward, 2 Backward, 3 Left, 4 Right, 5 Up, 6 Down
Gesture:   -1 Unknown, 0 Neutral, 1 Pinch, 2 Fist, 3 Open
Phase:     -1 Unknown, 0 Idle, 1 Onset, 2 Active, 3 Hold, 4 Release, 5 Transition
Consistency: -1 Unknown, 0 Normal, 1 Atypical
```

`consistency_score=-1` 表示条件样本不足，不能评价；其他值越接近 1 表示越偏离
训练数据中的同条件分布。第一版不得用 C 覆盖 D/H。

兼容地址 `/emgimu/state` 仍发送：

```text
timestamp_ms direction_as_legacy_gesture q_direction motion_intensity activation
```

旧协议无法表达手势、双阶段、一致性和信号质量，仅用于迁移期间维持旧前端。

