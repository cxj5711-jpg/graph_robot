#!/usr/bin/env python3
"""Thin real-gripper adapter for future Robotiq 2F-85 integration.

This file is intentionally not wired into the current task entry points yet.
It prepares a minimal real-robot gripper control layer that can be enabled
later without changing the high-level task logic.

Current assumptions:
- real robot is a UR controller reachable over TCP
- the Robotiq URCap helper functions will be used on the controller side
- when sending from an external Python process, the URCap function definitions
  usually need to be prepended to each remote script
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from robot_ip_client import RobotIPClient


class GripperClient:
    """Minimal real-gripper command adapter.

    Default behavior is fully disabled, so creating this client has no effect on
    the current codebase unless it is explicitly enabled and called later.
    """

    VALID_BACKENDS = {"disabled", "robotiq_urcap"}

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "disabled",
        host: Optional[str] = None,
        port: int = 30002,
        output_path: str = "robot_command_output/gripper_commands.txt",
        gripper_id: Optional[str] = None,
        definition_script_path: Optional[str] = None,
        prepend_definitions: bool = True,
        default_speed: int = 255,
        default_force: int = 128,
        command_client: Optional[RobotIPClient] = None,
        object_detected_reader: Optional[Callable[[], Optional[bool]]] = None,
    ):
        self.enabled = bool(enabled)
        self.backend = self._normalize_backend(backend)
        self.host = host
        self.port = int(port)
        self.output_path = output_path
        self.gripper_id = None if gripper_id is None else str(gripper_id)
        self.definition_script_path = definition_script_path
        self.prepend_definitions = bool(prepend_definitions)
        self.default_speed = self._clamp_byte(default_speed, "default_speed")
        self.default_force = self._clamp_byte(default_force, "default_force")
        self.object_detected_reader = object_detected_reader

        self._definition_script_cache: Optional[str] = None
        self._owns_command_client = command_client is None
        self.command_client = command_client or RobotIPClient(
            host=host,
            port=port,
            enabled=self.enabled,
            output_path=output_path,
            clear_on_connect=False,
        )

    def _normalize_backend(self, backend: str) -> str:
        normalized = str(backend).strip().lower()
        if normalized not in self.VALID_BACKENDS:
            raise ValueError(
                f"Unsupported gripper backend '{backend}'. "
                f"Expected one of {sorted(self.VALID_BACKENDS)}."
            )
        return normalized

    def _clamp_byte(self, value: int, name: str) -> int:
        ivalue = int(value)
        if not 0 <= ivalue <= 255:
            raise ValueError(f"{name} must be within [0, 255], got {value}.")
        return ivalue

    def _format_urcap_call(self, function_name: str, *args) -> str:
        arg_list = [self._format_script_arg(arg) for arg in args]
        if self.gripper_id is not None:
            arg_list.append(self._format_script_arg(self.gripper_id))
        return f"{function_name}({', '.join(arg_list)})"

    @staticmethod
    def _format_script_arg(value) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    def _load_definition_script(self) -> str:
        if self._definition_script_cache is not None:
            return self._definition_script_cache

        if not self.definition_script_path:
            self._definition_script_cache = ""
            return self._definition_script_cache

        path = os.path.abspath(self.definition_script_path)
        with open(path, "r", encoding="utf-8") as f:
            self._definition_script_cache = f.read().strip()
        return self._definition_script_cache

    def _send_command_lines(self, command_lines, description: str) -> Optional[str]:
        if not self.enabled or self.backend == "disabled":
            return None
        if self.backend != "robotiq_urcap":
            raise ValueError(f"Unsupported enabled backend '{self.backend}'.")

        command_lines = [str(line).strip() for line in command_lines if str(line).strip()]
        if not command_lines:
            return None

        definition_script = self._load_definition_script()
        if self.prepend_definitions and not definition_script:
            raise RuntimeError(
                "Robotiq remote control requires URCap function definitions, but "
                "definition_script_path is not configured."
            )

        if not getattr(self.command_client, "enabled", False):
            self.command_client.enabled = True
        if not getattr(self.command_client, "host", None):
            self.command_client.host = self.host
        if not getattr(self.command_client, "port", None):
            self.command_client.port = self.port

        script_parts = []
        if self.prepend_definitions and definition_script:
            script_parts.append(definition_script)
        script_parts.extend(command_lines)
        script = "\n".join(script_parts) + "\n"

        log_lines = [f"# gripper: {description}"]
        log_lines.extend(command_lines)
        return self.command_client.send_script(script, log_lines=log_lines)

    def connect(self):
        if self.enabled and self.command_client is not None:
            self.command_client.connect()

    def disconnect(self):
        if self.command_client is not None and self._owns_command_client:
            self.command_client.close()

    def activate(self, wait: bool = True) -> Optional[str]:
        command = "rq_activate_and_wait" if wait else "rq_activate"
        return self._send_command_lines([self._format_urcap_call(command)], "activate")

    def open(self, wait: bool = True) -> Optional[str]:
        command = "rq_open_and_wait" if wait else "rq_open"
        return self._send_command_lines([self._format_urcap_call(command)], "open")

    def close_gripper(self, wait: bool = True) -> Optional[str]:
        command = "rq_close_and_wait" if wait else "rq_close"
        return self._send_command_lines([self._format_urcap_call(command)], "close")

    def close(self, wait: bool = True) -> Optional[str]:
        if isinstance(wait, bool):
            return self.close_gripper(wait=wait)
        raise TypeError("close(wait=...) expects a boolean wait flag.")

    def move(
        self,
        position: int,
        speed: Optional[int] = None,
        force: Optional[int] = None,
        wait: bool = True,
    ) -> Optional[str]:
        position = self._clamp_byte(position, "position")
        speed = self.default_speed if speed is None else self._clamp_byte(speed, "speed")
        force = self.default_force if force is None else self._clamp_byte(force, "force")

        move_command = "rq_move_and_wait" if wait else "rq_move"
        command_lines = [
            self._format_urcap_call("rq_set_speed", speed),
            self._format_urcap_call("rq_set_force", force),
            self._format_urcap_call(move_command, position),
        ]
        return self._send_command_lines(command_lines, "move")

    def is_object_detected(self) -> Optional[bool]:
        if not self.enabled or self.backend == "disabled":
            return None
        if callable(self.object_detected_reader):
            return self.object_detected_reader()
        raise NotImplementedError(
            "Object detection feedback is not available over the current write-only "
            "socket path. Add a status reader later (e.g. RTDE or a gripper state hook)."
        )
