from cb2.tempo import *

th = TempoHistory.from_function(lambda t: 120 + 20*math.sin(t), domain_end=10)
print(th)
th.show_plot()
thenv = th.as_tempo_envelope()
print(thenv)
thenv.show_plot()

print(TempoHistory(100))
print(TempoHistory(100).as_tempo_envelope())

# th = TempoHistory([60, 13, 180], [2, 5], beat=3)
# th.advance_time(9)
# th.show_plot(units="rate")