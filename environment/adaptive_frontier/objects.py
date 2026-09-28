from enum import IntEnum

class ObjectType(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    RESOURCE = 3
    GATE = 4
    HAZARD = 5
    DYNAMIC_TERRAIN = 6
    AGENT = 7
