import zmq
import time
import struct
import numpy as np
from knowme import get_configuration, setup_annotation, get_sound_stream
from pocketsphinx import *
from sphinxbase import *

import sys, os
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

system_config = get_configuration()
stream_socket, req_socket = setup_annotation(system_config)


modeldir  = "/usr/local/lib/python3.5/dist-packages/pocketsphinx/model"

##### Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-us'))
config.set_string('-dict', os.path.join(modeldir, 'cmudict-en-us.dict'))
config.set_string('-keyphrase', 'microbes')
config.set_float('-kws_threshold', 1e+20)

##### Process audio chunk by chunk. On keyphrase detected perform action and restart search
decoder = Decoder(config)
decoder.start_utt()


last_timestamp = time.time()

while True:
    timestamp, samples = get_sound_stream(stream_socket)
    samples = np.asarray(samples, dtype=np.float32)
    i = np.iinfo(np.int16)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    data = (samples * abs_max + offset).clip(i.min, i.max).astype(np.int16)
    decoder.process_raw(data, False, False)
    if decoder.hyp() != None:
        print ([(seg.word, seg.prob, seg.start_frame, seg.end_frame) for seg in decoder.seg()])
        print ("Detected keyphrase, restarting search")
        decoder.end_utt()
        decoder.start_utt()
    #print("Length of samples: ", len(samples))
    #print("Time interval: ", timestamp - last_timestamp)
    last_timestamp = timestamp
    #print("Message received")

exit()
