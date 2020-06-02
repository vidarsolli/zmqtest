#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys
import logging
import numpy as np
import librosa
import json
from keras.models import model_from_json
from sklearn.cluster import KMeans
import pickle
import matplotlib.animation as animation
import zmq
import time
import struct
import numpy as np
from knowme import get_configuration, setup_sound, get_sound_stream




import time


q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            # check for a message, this will not block
            raw_message = sub_socket.recv(flags=zmq.NOBLOCK)
            samples = np.frombuffer(raw_message, dtype=np.float32, offset=14, count=-1)
            data = np.asarray(samples)
            data = np.reshape(data, (len(data), 1))

            i = np.iinfo(np.int16)
            abs_max = 2 ** (i.bits - 1)
            offset = i.min + abs_max
            data = (data * abs_max + offset).clip(i.min, i.max).astype(np.int16)

        except zmq.Again as e:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    #return lines, time_text, energy_text
    return lines



import matplotlib.pyplot as plt
import threading

# Setup the socket
SOUND_TOPIC = 'SOUND'
system_config = get_configuration()
context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(
    "tcp://" + system_config.communication.sensorIp + ":" + str(system_config.communication.sensorPort))
sub_socket.setsockopt_string(zmq.SUBSCRIBE, SOUND_TOPIC)

#length = int(args.window * args.samplerate / (1000 * args.downsample))
length = 2000
print("Length: ", length)
plotdata = np.zeros((length, 1))
sound_class = "No"

fig, ax = plt.subplots()
lines = ax.plot(plotdata)

ax.axis((0, len(plotdata), -32000, 32000))
ax.set_yticks([0])
ax.yaxis.grid(True)
ax.tick_params(bottom='off', top='off', labelbottom='off',
               right='off', left='off', labelleft='off')
fig.tight_layout(pad=0)
ani = animation.FuncAnimation(fig, update_plot, interval=200, blit=True)

plt.show()
