import zmq
import sys
import time
import numpy as np
from knowme import get_configuration
import sounddevice as sd
import queue

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    print("Data received")
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


SOUND_TOPIC = b"SOUND"

system_config = get_configuration()
print(system_config.__str__() + '\n')

context = zmq.Context()
stream_socket = context.socket(zmq.PUB)
stream_socket.bind("tcp://*:%d" % system_config.communication.sensorPort)

print(sd.query_devices())

#istream.close()
istream = sd.InputStream(samplerate=system_config.sound.sampleRate, device=9, channels=1, callback=audio_callback)


dummy_data = np.zeros((2000), dtype=np.uint16)
timestamp = np.zeros((1), dtype=np.float64)
#dummy_data[0] = 1234
#dummy_data[1] = 5678
while True:
    # Create an array of 16 bit random data
    timestamp[0] = time.time()
    print("Sending message")
    stream_socket.send(SOUND_TOPIC + b' ' + timestamp.tobytes() + dummy_data.tobytes())
    print(b'timestamp')
    time.sleep(1)




exit()
