""" cb2 test package. Installs suite-wide time compression on import (no-op unless CB2_TEST_COMPRESSION is set). """
from tests.timing import install

install()
