"""Type definitions and enumerations used across the KWS package."""

from enum import IntEnum


class LabelIndex(IntEnum):
    """Label indices for special categories."""

    SILENCE_INDEX = 0
    UNKNOWN_WORD_INDEX = 1
