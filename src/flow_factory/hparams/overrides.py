# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for parsing and applying dot-style CLI overrides."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import yaml


def parse_cli_overrides(tokens: list[str]) -> dict[str, Any]:
    """Parse CLI override tokens into a mapping.

    Supported forms:
    - key=value
    - --key=value
    - --key value

    Notes:
    - Last value wins for repeated keys.
    - Dot-path list indexing (e.g. rewards.0.weight) is intentionally unsupported.
    """
    overrides: dict[str, Any] = {}
    idx = 0

    while idx < len(tokens):
        key, value_token, consumed = _parse_override_token(tokens, idx)
        _validate_override_key(key)
        overrides[key] = _parse_override_value(value_token, key)
        idx += consumed

    return overrides


def apply_dot_overrides(config: dict[str, Any] | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply dot-style overrides to a nested config mapping."""
    if config is None:
        merged: dict[str, Any] = {}
    elif isinstance(config, dict):
        merged = copy.deepcopy(config)
    else:
        raise ValueError(f"YAML root must be a mapping, got {type(config).__name__}.")

    for key, value in overrides.items():
        _set_nested_value(merged, key, value)

    return merged


def _parse_override_token(tokens: list[str], idx: int) -> tuple[str, str, int]:
    token = tokens[idx]

    if token.startswith("--"):
        raw = token[2:]
        if not raw:
            raise ValueError("Found bare '--' in override arguments.")

        if "=" in raw:
            key, value = raw.split("=", 1)
            return key, value, 1

        if idx + 1 >= len(tokens):
            raise ValueError(f"Missing value for override '--{raw}'.")

        value = tokens[idx + 1]
        if value.startswith("--"):
            raise ValueError(f"Missing value for override '--{raw}'.")

        return raw, value, 2

    if token.startswith("-"):
        raise ValueError(
            f"Unsupported option '{token}'. Use one of: key=value, --key=value, --key value."
        )

    if "=" in token:
        key, value = token.split("=", 1)
        return key, value, 1

    raise ValueError(
        f"Unsupported argument '{token}'. Use one of: key=value, --key=value, --key value."
    )


def _parse_override_value(value_token: str, key: str) -> Any:
    if value_token == "":
        return ""

    try:
        return yaml.safe_load(value_token)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML value for override '{key}': {exc}") from exc


def _validate_override_key(key: str) -> None:
    if not key:
        raise ValueError("Override key cannot be empty.")

    if key.startswith(".") or key.endswith(".") or ".." in key:
        raise ValueError(f"Invalid override key '{key}'.")

    parts = key.split(".")
    for part in parts:
        if not part:
            raise ValueError(f"Invalid override key '{key}'.")
        if any(c.isspace() for c in part):
            raise ValueError(f"Override key '{key}' cannot contain whitespace.")
        if part.isdigit() or "[" in part or "]" in part:
            raise ValueError(
                f"List indexing is not supported in override key '{key}'. "
                "Pass full list values instead."
            )


def _set_nested_value(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor: dict[str, Any] = target

    for index, part in enumerate(parts[:-1]):
        current = cursor.get(part)

        if current is None:
            cursor[part] = {}
            current = cursor[part]
        elif not isinstance(current, dict):
            prefix = ".".join(parts[: index + 1])
            raise ValueError(
                f"Cannot apply override '{key}': '{prefix}' is not a mapping in current config."
            )

        cursor = current

    cursor[parts[-1]] = value
