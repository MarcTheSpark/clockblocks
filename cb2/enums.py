from enum import Enum


class DurationUnits(str, Enum):
    BEATS = "beats"
    TIME = "time"


class TempoUnits(str, Enum):
    TEMPO = "tempo"
    RATE = "rate"
    BEATLENGTH = "beatlength"
