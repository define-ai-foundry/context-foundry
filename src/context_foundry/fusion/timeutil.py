# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone


def as_utc(moment: datetime) -> datetime:
    """Treat a naive datetime as UTC; convert an aware one to UTC.

    Sources differ on whether they carry tzinfo, and CoT timestamps are labelled
    `Z`. Normalising here keeps that label honest instead of asserting UTC over
    whatever offset the input happened to be in.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
