# import random
# import time
#
#
# def yielding_routine():
#     while True:
#         print("hello")
#         yield random.choice([0.5, 1.0])
#
#
#
# # Scheduler
# yr = yielding_routine()
# delay = next(yr)
# while True:
#     time.sleep(delay)
#     delay = next(yr)
import inspect
from inspect import currentframe
from functools import partial

def wait_wrapper(func):
    def wrapped_func(yield_amount=None):
        if yield_amount:
            yield yield_amount




def yielding_routine():
    print("hello")
    wait(2)
    print("goodbye")

# wait_wrapper(yielding_routine)

import inspect
import re
import ast

exec(ast.unparse(ast.parse(inspect.getsource(yielding_routine)))

print()