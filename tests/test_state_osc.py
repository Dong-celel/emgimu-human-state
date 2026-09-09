import unittest

from emgimu.osc import OSC_V2_ADDRESS, decode_message, encode_message
from emgimu.service import RAW_OSC_ADDRESS, parse_raw_message
from emgimu.state import (
    Confidence, Consistency, Direction, Gesture, HumanState, Phase, PhasePair,
    QualityFlag, SignalQuality,
)


class StateOscTests(unittest.TestCase):
    def test_v2_round_trip(self):
        state = HumanState(
            2**40, Direction.RIGHT, Gesture.FIST, 0.7,
            PhasePair(Phase.ACTIVE, Phase.HOLD), 0.4,
            Consistency.NORMAL, Confidence(0.9, 0.8, 0.7, 0.6),
            SignalQuality(0.95, 0.96, QualityFlag.TIMESTAMP_VALID | QualityFlag.EMG_VALID | QualityFlag.IMU_VALID),
            0.1,
        )
        address, args = decode_message(encode_message(OSC_V2_ADDRESS, *state.osc_v2_args()))
        self.assertEqual(address, OSC_V2_ADDRESS)
        self.assertEqual(args[0], 2**40)
        self.assertEqual(args[1:5], [int(Direction.RIGHT), int(Gesture.FIST), int(Phase.ACTIVE), int(Phase.HOLD)])
        self.assertEqual(len(args), 16)

    def test_unknown_maps_to_none_only_in_legacy(self):
        state = HumanState(
            1, Direction.UNKNOWN, Gesture.OPEN, 0, PhasePair(Phase.UNKNOWN, Phase.HOLD), 0,
            Consistency.UNKNOWN, Confidence(0, 1, 0, 1), SignalQuality(1, 1),
        )
        self.assertEqual(state.legacy_args()[1], 0)
        self.assertEqual(state.osc_v2_args()[1], -1)

    def test_raw_input_packet(self):
        packet = encode_message(RAW_OSC_ADDRESS, 100, *[float(index) for index in range(14)])
        sample = parse_raw_message(packet)
        self.assertEqual(sample.timestamp_ms, 100)
        self.assertEqual(sample.emg.shape, (8,))
        self.assertEqual(sample.accel.shape, (3,))
        self.assertEqual(sample.gyro.shape, (3,))


if __name__ == "__main__":
    unittest.main()
