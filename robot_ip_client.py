from __future__ import annotations

import os
import socket
from typing import Iterable, Optional

import numpy as np


class RobotIPClient:
    """
    Thin execution adapter for robot commands.

    Current stage:
    - keep the API close to the earlier file-based adapter
    - optionally send URScript-style command strings over TCP
    - optionally mirror the same command stream to a local text log
    """

    VALID_SEND_MODES = {"solver_iteration", "linear_step", "solver_final"}

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        a: float = 0.15,
        v: float = 0.08,
        enabled: bool = False,
        output_path: str = "robot_command_output/movej_commands.txt",
        send_mode: str = "linear_step",
        clear_on_connect: bool = True,
        socket_timeout: float = 3.0,
    ):
        self.host = host
        self.port = None if port is None else int(port)
        self.a = float(a)
        self.v = float(v)
        self.enabled = bool(enabled)
        self.output_path = output_path
        self.send_mode = send_mode
        self.clear_on_connect = bool(clear_on_connect)
        self.socket_timeout = float(socket_timeout)
        self.connected = False
        self.sock: Optional[socket.socket] = None
        self.last_command: Optional[str] = None
        self.enabled_stages = self._normalize_send_modes(send_mode)

    def _normalize_send_modes(self, send_mode) -> set[str]:
        if isinstance(send_mode, str):
            modes = [send_mode]
        else:
            modes = list(send_mode)

        normalized = {str(mode) for mode in modes}
        invalid = normalized.difference(self.VALID_SEND_MODES)
        if invalid:
            raise ValueError(
                f"Unsupported send_mode value(s) {sorted(invalid)}. "
                f"Expected values from {sorted(self.VALID_SEND_MODES)}."
            )
        return normalized

    def stage_enabled(self, stage: Optional[str]) -> bool:
        if stage is None:
            return True
        return stage in self.enabled_stages

    def _prepare_output_file(self):
        if not self.output_path:
            return

        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        mode = "w" if self.clear_on_connect else "a"
        with open(self.output_path, mode, encoding="utf-8"):
            pass

    def connect(self):
        if not self.enabled:
            return

        self._prepare_output_file()

        if self.host and self.port and self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.socket_timeout)

        self.connected = True

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
        self.connected = False

    def format_movej(
        self,
        joint_angles: Iterable[float],
        a: Optional[float] = None,
        v: Optional[float] = None,
        t: Optional[float] = None,
        r: Optional[float] = None,
    ) -> str:
        values = np.asarray(list(joint_angles), dtype=float).reshape(-1)
        if values.size != 6:
            raise ValueError(f"Expected 6 joint angles, got {values.size}.")

        accel = self.a if a is None else float(a)
        speed = self.v if v is None else float(v)
        joints = ",".join(f"{val:.6f}" for val in values)
        parts = [f"[{joints}]", f"a={accel:.6f}", f"v={speed:.6f}"]
        if t is not None:
            parts.append(f"t={float(t):.6f}")
        if r is not None:
            parts.append(f"r={float(r):.6f}")
        return f"movej({', '.join(parts)})"

    def emit_joint_command(self, joint_angles: Iterable[float], stage: Optional[str] = None) -> Optional[str]:
        if not self.stage_enabled(stage):
            return None
        return self.send_movej(joint_angles)

    def _write_output_lines(self, lines: Iterable[str]):
        if not self.output_path:
            return
        with open(self.output_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")

    def send_script(self, script: str, log_lines: Optional[Iterable[str]] = None) -> Optional[str]:
        if not self.enabled:
            return None
        if not self.connected:
            self.connect()

        normalized = script if script.endswith("\n") else script + "\n"
        payload = normalized.encode("utf-8")

        if self.sock is not None:
            self.sock.sendall(payload)

        if log_lines is None:
            self._write_output_lines(normalized.splitlines())
        else:
            self._write_output_lines(log_lines)
        self.last_command = normalized.rstrip()
        return normalized.rstrip()

    def format_urscript_program(self, commands: Iterable[str], program_name: str = "remote_motion") -> str:
        safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in program_name).strip("_")
        if not safe_name:
            safe_name = "remote_motion"

        command_list = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        if not command_list:
            raise ValueError("Cannot build a URScript program with no commands.")

        lines = [f"def {safe_name}():"]
        lines.extend(f"  {cmd}" for cmd in command_list)
        lines.append("end")
        return "\n".join(lines) + "\n"

    def send_program(self, commands: Iterable[str], program_name: str = "remote_motion") -> Optional[str]:
        command_list = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        if not command_list:
            return None

        script = self.format_urscript_program(command_list, program_name=program_name)
        return self.send_script(script, log_lines=command_list)

    def send_movej(
        self,
        joint_angles: Iterable[float],
        a: Optional[float] = None,
        v: Optional[float] = None,
        t: Optional[float] = None,
        r: Optional[float] = None,
    ) -> Optional[str]:
        command = self.format_movej(joint_angles, a=a, v=v, t=t, r=r)
        self.send_script(command)
        self.last_command = command
        return command
