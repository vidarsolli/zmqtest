import sys
import zmq
import numpy as np
import KnowMeAiConfig_pb2

SOUND_TOPIC = 'SOUND'

def get_configuration():
    configIp = "localhost"
    configPort = "5580"
    if len(sys.argv) > 1:
        configIp = sys.argv[1]

    # Socket to talk to server
    context = zmq.Context()
    configService = context.socket(zmq.REQ)

    configService.connect("tcp://" + configIp + ":" + configPort)
    print ("Sending request for configuration data")
    configService.send_string(sys.argv[0])
    # Wait for the configuration message
    print("Waiting for configuration message")
    msg = configService.recv()
    print("Configuratin message received")
    systemConfig = KnowMeAiConfig_pb2.systemConfig()
    systemConfig.ParseFromString( msg )
    return systemConfig

def setup_sound(system_conf):
    context = zmq.Context()
    # Setup the subscription socket
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(
        "tcp://" + system_conf.communication.sensorIp + ":" + str(system_conf.communication.sensorPort))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, SOUND_TOPIC)
    # Setup the publishing socket
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:" + str(system_conf.communication.soundPort))

    return sub_socket, pub_socket

def get_sound_stream(sub_socket):
    raw_message = sub_socket.recv()
    timestamp = np.frombuffer(raw_message, dtype=np.float64, offset= 6, count=1)
    samples = np.frombuffer(raw_message, dtype=np.int16, offset=14, count=-1)

    return timestamp, samples

def setup_annotation(system_conf):
    context = zmq.Context()
    # Setup the subscription socket
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(
        "tcp://" + system_conf.communication.sensorIp + ":" + str(system_conf.communication.sensorPort))
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, SOUND_TOPIC)
    # Setup the publishing socket
    req_socket = context.socket(zmq.REQ)
    req_socket.connect("tcp://" + system_conf.communication.sensorIp + ":" + str(system_conf.communication.storePort))

    return sub_socket, req_socket
