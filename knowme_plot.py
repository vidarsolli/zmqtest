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



import time



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
args = parser.parse_args()
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()
q2 = queue.Queue()
q3 = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])
    q2.put(indata[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    #return lines, time_text, energy_text
    return lines

def animate(i):
    """perform animation step"""
    global sound_class
    time_text.set_text('time = %.1f' % i)
    energy_text.set_text('energy = %s J' % sound_class)
    return energy_text


def classify_sound(sound_queue, cp, encoder, cluster, labels):
    global sound_class
    short_term_length = int(cp["short_term"] * cp["sample_rate"])
    step_length = int(cp["step_size"]*cp["sample_rate"])
    no_of_short_terms = int(cp["mid_term"]/cp["step_size"])
    mfcc_buffer = np.zeros((no_of_short_terms, cp["n_mfcc"]))
    input = np.zeros((1, no_of_short_terms, cp["n_mfcc"], 1))
    a = np.array([])
    mfcc = np.zeros((cp["n_mfcc"], 3))
    sound_samples = np.zeros((short_term_length))

    print("Starting classify thread", sound_queue)
    while True:
        while not sound_queue.empty():
            a = np.append(a, sound_queue.get())
            j = 0
            k = 0
            # Loop through the sound samples in step_size and remove the used part
            while (a.shape[0] - j*step_length) > (short_term_length + step_length):
                sound_samples = a[j * step_length:(j * step_length+short_term_length)]
                j += 1
                #print(j, len(a))
                mfcc = librosa.feature.mfcc(y=sound_samples, sr=cp["sample_rate"],\
                                                      hop_length=step_length, window='hann', n_mfcc=cp["n_mfcc"])
                mfcc_buffer[k,:] = mfcc[:,0]
                k=(k+1)%no_of_short_terms
                # Make the input for prediction
                l=k
                for i in range(no_of_short_terms):
                    input[0,i,:,0] = mfcc_buffer[l,:]
                    l=(l+1)%no_of_short_terms
                #input = mfcc_buffer[:,k:]+mfcc_buffer[:,k]
                # Predict the class
                #print(mfcc.shape, mfcc_buffer.shape, input.shape)
                # evaluate loaded model on test data
                score = encoder.predict(input)
                result = cluster.predict(score)
                q3.put(result)
                sound_class = labels[result]
            a = a[j * step_length:]

try:
    import matplotlib.pyplot as plt
    import sounddevice as sd
    import threading

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with open("classify_sound.json") as file:
        cp = json.load(file)

    # Load the models
    json_file = open('../ClassifySound/encoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json)
    # load weights into new model
    encoder.load_weights("../ClassifySound/encoder.h5")
    #encoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    cluster = pickle.load(open('../ClassifySound/kmeans_model.sav', 'rb'))
    labels = np.load('../ClassifySound/sorted_labels.npy', allow_pickle=True)
    print(labels)
    print("Loaded model from disk")

    # Create the thread that will handle sound classification
    logging.info("Creating the classify thread")
    x = threading.Thread(target=classify_sound, args=(q2,cp,encoder,cluster,labels))
    x.start()

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    sound_class = "No"

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)

    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom='off', top='off', labelbottom='off',
                   right='off', left='off', labelleft='off')
    fig.tight_layout(pad=0)
    print("Device : ", args.device, args.samplerate)
    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = animation.FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
    #                    xlim=(-2, 2), ylim=(-2, 2))
    ax2 = fig2.add_subplot(111)
    #ax2.grid()
    time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
    energy_text = ax2.text(0.02, 0.90, '', transform=ax2.transAxes)
    ani2 = animation.FuncAnimation(fig2, animate, interval=args.interval, blit=True)

    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))