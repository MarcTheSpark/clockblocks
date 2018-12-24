from clockblocks import Clock

clock = Clock(initial_tempo=60)


def log_timing():
    print("Beat: {}, Time: {}, Current Tempo: {}".format(clock.beats(), clock.time(), round(clock.tempo, 2)))


while clock.beats() < 4:
    log_timing()
    clock.wait(1)

# sudden change to 120 bpm (i.e. 2 beats per second)
clock.tempo = 120

while clock.beats() < 8:
    log_timing()
    clock.wait(1)

# gradually slow to 30 bpm over the course of 8 beats
clock.set_tempo_target(30, 8)

while clock.beats() < 20:
    log_timing()
    clock.wait(1)

clock.tempo_envelope.show_plot(resolution=200, title="Example TempoEnvelope", units="tempo")