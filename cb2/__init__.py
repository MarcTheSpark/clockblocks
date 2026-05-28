from importlib.metadata import version, PackageNotFoundError

from cb2.time_stamp import TimeStamp

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass