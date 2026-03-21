from enum import Enum


class DecisionEnum(str, Enum):
    yes = "yes"
    no = "no"
    maybe = "maybe"
    unknown = "unknown"