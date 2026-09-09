from __future__ import annotations

import socket
import struct
import threading
from typing import Any

from .state import HumanState


OSC_V2_ADDRESS = "/emgimu/state/v2"
OSC_LEGACY_ADDRESS = "/emgimu/state"


class OscError(ValueError):
    pass


def _pad4(length: int) -> int:
    return (4 - length % 4) % 4


def _string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + b"\x00" * _pad4(len(raw))


def encode_message(address: str, *args: Any) -> bytes:
    if not address.startswith("/"):
        raise OscError("OSC address must start with '/'")
    tags: list[str] = []
    payload = bytearray()
    for arg in args:
        if isinstance(arg, bool):
            tags.append("i")
            payload.extend(struct.pack(">i", int(arg)))
        elif isinstance(arg, int):
            if -(2**31) <= arg < 2**31:
                tags.append("i")
                payload.extend(struct.pack(">i", arg))
            else:
                tags.append("h")
                payload.extend(struct.pack(">q", arg))
        elif isinstance(arg, float):
            tags.append("f")
            payload.extend(struct.pack(">f", arg))
        elif isinstance(arg, str):
            tags.append("s")
            payload.extend(_string(arg))
        else:
            raise OscError(f"unsupported OSC argument type: {type(arg)!r}")
    return _string(address) + _string("," + "".join(tags)) + bytes(payload)


def decode_message(data: bytes) -> tuple[str, list[int | float | str]]:
    def read_string(offset: int) -> tuple[str, int]:
        try:
            end = data.index(0, offset)
        except ValueError as exc:
            raise OscError("unterminated OSC string") from exc
        text = data[offset:end].decode("utf-8")
        consumed = end - offset + 1
        return text, end + 1 + _pad4(consumed)

    address, offset = read_string(0)
    tags, offset = read_string(offset)
    if not tags.startswith(","):
        raise OscError("invalid OSC type tags")
    output: list[int | float | str] = []
    for tag in tags[1:]:
        sizes = {"i": 4, "h": 8, "f": 4}
        if tag == "s":
            value, offset = read_string(offset)
            output.append(value)
            continue
        if tag not in sizes or offset + sizes[tag] > len(data):
            raise OscError(f"invalid or truncated OSC value {tag!r}")
        fmt = {"i": ">i", "h": ">q", "f": ">f"}[tag]
        output.append(struct.unpack_from(fmt, data, offset)[0])
        offset += sizes[tag]
    return address, output


class OscPublisher:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        *,
        publish_legacy: bool = False,
    ) -> None:
        self.target = (host, int(port))
        self.publish_legacy = bool(publish_legacy)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.lock = threading.Lock()

    def publish(self, state: HumanState) -> None:
        packets = [encode_message(OSC_V2_ADDRESS, *state.osc_v2_args())]
        if self.publish_legacy:
            packets.append(encode_message(OSC_LEGACY_ADDRESS, *state.legacy_args()))
        with self.lock:
            for packet in packets:
                self.socket.sendto(packet, self.target)

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> "OscPublisher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
