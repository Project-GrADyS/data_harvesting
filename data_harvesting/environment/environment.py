import enum


class EndCause(enum.Enum):
    NONE = 0
    TIMEOUT = 1
    ALL_COLLECTED = 2
    STALLED = 3
