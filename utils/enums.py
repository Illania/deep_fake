from enum import Enum


class SourceType(Enum):
    IMAGE = 0
    VIDEO = 1
    ARCHIVE = 2


class HashAlgorithm(Enum):
    SHA1 = 0
    SHA256 = 1
