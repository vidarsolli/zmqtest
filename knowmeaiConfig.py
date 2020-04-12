import zmq
import random
import sys
import time
import KnowMeAiConfig_pb2

configMsg = KnowMeAiConfig_pb2.systemConfig()
configMsg.sound.sampleRate = 8000
configMsg.sound.samples = 4000
configMsg.skeleton.noOfPoints = 40
configMsg.skeleton.frames = 15
configMsg.face.width = 32
configMsg.face.height = 32
configMsg.face.frames = 15
configMsg.face.color = KnowMeAiConfig_pb2.systemConfig.colorType.RGB
configMsg.communication.sensorIp = "localhost"
configMsg.communication.soundIp = "localhost"
configMsg.communication.faceIp = "localhost"
configMsg.communication.gestureIp = "localhost"
configMsg.communication.syncIp = "localhost"
configMsg.communication.compoundIp = "localhost"
configMsg.communication.sensorPort = 55555;
configMsg.communication.soundPort = 55556;
configMsg.communication.facePort = 55557;
configMsg.communication.gesturePort = 55558;
configMsg.communication.storePort = 55575;
configMsg.communication.syncPort = 55560;
configMsg.communication.compoundPort = 55570;



if configMsg.IsInitialized():
    print("Config message is initialized")
else:
    print("Config message not initialized correctly")
    exit()

# Establish the zmq socket
port = "5580"
context = zmq.Context()
configService = context.socket(zmq.REP)
configService.bind("tcp://*:%s" % port)

# Build the configuration message
configMessage = configMsg.SerializeToString()

while True:
    # Wait for a request message
    print("Waiting for request")
    msg = configService.recv()
    print("Configuration request received from ", str(msg))
    # Respond with the configuration message
    configService.send(configMessage)
