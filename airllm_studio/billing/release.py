# Copyright 2026 Michael Beck. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release vs development build detection for Dev Unlock."""

from __future__ import annotations

import os


def is_release_build() -> bool:
    """True for customer / MAS builds where Dev Unlock must stay off.

    Set AIRLLM_STUDIO_RELEASE=1 in the .app launcher when signing for
    Mac App Store or other Release distribution. Ad-hoc and local
    developer runs leave it unset so QA can unlock Pro without StoreKit.
    """
    flag = (os.environ.get("AIRLLM_STUDIO_RELEASE") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def dev_unlock_available() -> bool:
    """Dev Unlock Pro is only offered outside Release builds."""
    return not is_release_build()
