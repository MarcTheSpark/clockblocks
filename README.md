# clockblocks

_clockblocks_ is a python library for controlling the flow of time, designed with musical applications in mind. In particular, it is a central component of [SCAMP](https://github.com/MarcTheSpark/scamp)  (**S**uite for **C**omputer-**A**ssisted **M**usic in **P**ython).

A `Clock` acts like a thread, but with the advantage that when multiple clocks are coordinated under the same master clock they remain precisely coordinated and do not experience drift. Furthermore, processing time is taken into account when "wait" is called in a given Clock. For example, the following program:

```python
import clockblocks
import time
import math

clock = clockblocks.Clock()
start = time.time()

while True:
    print("Current time: {}".format(round(time.time() - start, 4)))
    # do some pointless and time-consuming calculations
    for i in range(1000000):
        math.log((i+1)**0.7)
    clock.wait(2)
```
 
... generates the output:

```console
Current time: 0.0
Current time: 2.0001
Current time: 4.0001
Current time: 6.0
Current time: 8.0001
Current time: 10.0
```

Whereas a traditional thread:

```python
import time
import math

start = time.time()

while True:
    print("Current time: {}".format(round(time.time() - start, 4)))
    # do some pointless and time-consuming calculations
    for i in range(1000000):
        math.log((i+1)**0.7)
    time.sleep(2)
```


...will gradually drift because of the intensive calculations, outputting:

```console
Current time: 0.0
Current time: 2.3772
Current time: 4.7623
Current time: 7.1397
Current time: 9.5151
Current time: 11.893
```

In addition, _clockblocks_ offers useful musical functionality, like sudden and gradual changes of tempo, the concept of a `Moment` that can be defined either in terms of beats or wall time, and the related ability to control metric phase. 

Perhaps the most exciting feature of _clockblocks_ is that clocks moving at different tempi remain coordinated and can even be nested within each other. In this case, each clock distorts time for those underneath it: a clock whose tempo is oscillating between slow and fast, nested within a clock that is accelerating, will generate a time stream whose tempo oscillates between faster and faster values.

## The Version 1.0 Rewrite

The initial 0.x version of _clockblocks_ was implemented in a way that aligned conceptually with the API: each clock
registered its wake up times with its parent recursively. However, practically this turned out to be both unduly 
complicated and limiting, especially in situations with live interaction or external synchronization.

Version 1.x is a ground-up rewrite of clockblocks around a single background `Scheduler`. The concepts above are all
preserved — `Clock`, `wait`, tempo changes, and nested clocks work as shown — but 1.0 is **not** a drop-in upgrade:
several APIs were dropped or reshaped. The one most existing code will hit is that a forked function no longer receives
its clock as an argument — call `current_clock()` inside it instead. See the [CHANGELOG](CHANGELOG.md) for the full list
of breaking changes and a migration table from 0.6.x. The previous 0.x implementation lives on the
[`0.6.x` branch](https://github.com/MarcTheSpark/clockblocks/tree/0.6.x).
