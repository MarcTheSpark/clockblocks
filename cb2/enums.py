from enum import StrEnum


class DurationUnits(StrEnum):
    BEATS = "beats"
    TIME = "time"


class TempoUnits(StrEnum):
    TEMPO = "tempo"
    RATE = "rate"
    BEATLENGTH = "beatlength"
