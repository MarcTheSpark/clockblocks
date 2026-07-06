""" clockblocks test package. Installs suite-wide time compression on import (no-op unless CLOCKBLOCKS_TEST_COMPRESSION is set). """
from tests.timing import install

install()
