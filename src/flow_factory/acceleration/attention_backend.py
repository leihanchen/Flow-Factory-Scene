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

# src/flow_factory/acceleration/attention_backend.py
"""Attention-backend accelerator — the single code path that selects the
diffusers attention backend for every transformer.

This replaces the old ``BaseAdapter._set_attention_backend`` call and the
``model.attn_backend`` knob: the backend is now requested as an ``attention_backend``
entry in the acceleration ``shared`` list and applied here (after
``accelerator.prepare`` / ``post_init`` and before compile), so all transformer-level
acceleration flows through the same plugin mechanism.

The backend name is taken from the required ``backend`` param and forwarded to
diffusers' ``set_attention_backend`` verbatim — including approximate backends like
``sage`` — matching the previous behavior.

Marked ``stage='both'`` / ``safety='lossless'``: a backend is applied to the
transformer shared by rollout ``inference()`` and training ``forward()``, so the
two stay consistent (the property the validator's lossless category guarantees)
even for approximate kernels and coupled algorithms.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


def _is_hf_hub_offline() -> bool:
    """Return True when Hugging Face Hub offline mode is enabled."""
    val = os.getenv("HF_HUB_OFFLINE")
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_flash_attn3_revision() -> Optional[str]:
    """Resolve flash-attn3 revision from env or local cache refs."""
    for env_name in ["FLOW_FACTORY_FLASH_ATTN3_REVISION", "FLASH_ATTN3_REVISION"]:
        val = os.getenv(env_name)
        if val:
            return val.strip()

    cache_candidates: list[str] = []
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        cache_candidates.append(hf_hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_candidates.append(os.path.join(hf_home, "hub"))

    cache_candidates.append(os.path.expanduser("~/.cache/huggingface/hub"))

    rel_ref_path = os.path.join(
        "models--kernels-community--flash-attn3",
        "refs",
        "main",
    )
    for base in cache_candidates:
        ref_path = os.path.join(base, rel_ref_path)
        if os.path.isfile(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                if val:
                    return val
            except OSError:
                continue

    return None


def _pin_flash_attn3_hub_revision_for_offline(backend: str, *, is_main_process: bool) -> None:
    """Pin a concrete flash-attn3 revision so offline hub backends skip `/refs` lookups."""
    if backend not in {"_flash_3_hub", "_flash_3_varlen_hub"}:
        return

    if not _is_hf_hub_offline():
        return

    revision = _resolve_flash_attn3_revision()
    if not revision:
        if is_main_process:
            logger.warning(
                "HF_HUB_OFFLINE=1 and flash-attn3 hub backend is enabled, "
                "but no local revision was found. Set FLOW_FACTORY_FLASH_ATTN3_REVISION "
                "or ensure cache refs exist in HF_HUB_CACHE/HF_HOME."
            )
        return

    try:
        from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY

        patched = 0
        for cfg in _HUB_KERNELS_REGISTRY.values():
            if getattr(cfg, "repo_id", None) == "kernels-community/flash-attn3":
                cfg.revision = revision
                cfg.version = None
                patched += 1

        if is_main_process and patched > 0:
            logger.info(
                "Pinned flash-attn3 hub kernels to revision %s for offline mode "
                "(patched %d backend entries).",
                revision,
                patched,
            )
    except Exception as e:
        if is_main_process:
            logger.warning("Failed to pin flash-attn3 hub revision for offline mode: %s", e)


class AttentionBackendAccelerator(BaseAccelerator):
    """Set the diffusers attention backend on every transformer component.

    Parameters (from the entry's ``params``):
        backend: Backend name forwarded to ``transformer.set_attention_backend``
            (e.g. ``native`` / ``flash`` / ``_flash_3`` / ``_flash_3_hub`` /
            ``sage`` / ``xformers``). Required.

    See https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
    for the full list of supported backends.
    """

    safety = "lossless"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        backend = self.params.get("backend")
        if not backend:
            raise ValueError(
                "AttentionBackendAccelerator requires a `backend` param, e.g. "
                "`{ name: attention_backend, params: { backend: _flash_3_hub } }`."
            )

        _pin_flash_attn3_hub_revision_for_offline(
            str(backend),
            is_main_process=adapter.accelerator.is_main_process,
        )

        applied = False
        for name in adapter.transformer_names:
            transformer = adapter.get_component(name)
            if hasattr(transformer, "set_attention_backend"):
                transformer.set_attention_backend(backend)
                applied = True
                if adapter.accelerator.is_main_process:
                    logger.info(
                        "AttentionBackendAccelerator: set backend '%s' for '%s'.", backend, name
                    )
        if not applied:
            raise ValueError(
                f"AttentionBackendAccelerator: backend '{backend}' requested but none of the "
                f"adapter's transformer components {adapter.transformer_names} support "
                "`set_attention_backend`. Models with a custom attention implementation (e.g. "
                "Bagel, which forces flash_attention_2 at load) must not use this accelerator; "
                "remove the `attention_backend` entry from the acceleration config."
            )
