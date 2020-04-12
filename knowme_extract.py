import zmq
import sys
from datetime import datetime
import time
import soundfile as sf
import numpy as np
from knowme import get_configuration
import sounddevice as sd
import queue
from numpy_ringbuffer import RingBuffer

q = queue.Queue()
BUFFER_LENGTH = 10
DATA_SIZE = 8000
buffer = np.empty((BUFFER_LENGTH, DATA_SIZE), dtype = np.float32)
buffer_idx = 0
data_idx = 0

def sound_buffer_append(data):
    global buffer_idx
    global data_idx
    global buffer
    size = len(data)
    left = DATA_SIZE - data_idx
    if (left) < size:
        buffer[buffer_idx][data_idx:DATA_SIZE] = list(data[0:left])
        buffer_idx = (buffer_idx +1) % BUFFER_LENGTH
        buffer[buffer_idx][0:(size-left)]=list(data[left:])
        data_idx = size - left
    else:
        buffer[buffer_idx][data_idx:(data_idx+size)] = list(data)
        data_idx = data_idx + size
    print (size, buffer_idx, data_idx)

def sound_buffer_save():
    global buffer_idx
    global data_idx
    global buffer
    a = np.array([])
    idx = (buffer_idx + 1) % BUFFER_LENGTH
    for i in range(BUFFER_LENGTH - 1):
        a = np.append(a, buffer[idx])
        idx = (idx + 1)%BUFFER_LENGTH
    #a = np.reshape(a, (int(a.shape[0]/cp["audio_channels"]), cp["audio_channels"]))
    t = datetime.now()
    filenameExt = str(t.year) + "-" + str(t.month) + "-" + str(t.day) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + \
                  str(t.second)
    filename = filenameExt + ".wav"
    sfile = sf.SoundFile(filename, mode='x', samplerate=8000, channels=1, subtype="PCM_24")

    sfile.write(a)
    time.sleep(1)
    sfile.close()




def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    sound_buffer_append(indata.copy())
    q.put(indata.copy())


SOUND_TOPIC = b"SOUND"

system_config = get_configuration()
print(system_config.__str__() + '\n')

context = zmq.Context()
stream_socket = context.socket(zmq.PUB)
stream_socket.bind("tcp://*:%d" % system_config.communication.sensorPort)

print(sd.query_devices())

#istream.close()
istream = sd.InputStream(samplerate=system_config.sound.sampleRate, device=None, channels=1, callback=audio_callback)
istream.start()

timestamp = np.zeros((1), dtype=np.float64)
last_timestamp = time.time()
while True:
    data = q.get()
    # Create an array of 16 bit random data
    timestamp[0] = time.time()
    stream_socket.send(SOUND_TOPIC + b' ' + timestamp.tobytes() + data.tobytes())
    if (timestamp[0] - last_timestamp) > 10:
        print("saving sound_buffer")
        sound_buffer_save()
        last_timestamp = timestamp[0]
    #time.sleep(1)




exit()
