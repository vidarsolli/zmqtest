import zmq
import time
import struct
import numpy as np
from knowme import get_configuration, setup_annotation, get_sound_stream

system_config = get_configuration()
stream_socket, req_socket = setup_annotation(system_config)

last_timestamp = time.time()

while True:
    timestamp, samples = get_sound_stream(stream_socket)
    print("Length of samples: ", len(samples))
    print("Time interval: ", timestamp - last_timestamp)
    last_timestamp = timestamp
    print("Message received")

exit()
