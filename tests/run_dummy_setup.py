import os,sys
from ublarcvserver import ublarcvserver

"""
This script is used to test the Majordomo classes.

We implement a dummy setup where the client and worker just say hello to each other.
Also servers as an example.

We also setup the basic larcv client and worker, which pass larcv images back and forth.
"""

# endpoint:
endpoint  = "tcp://localhost:6002"
bindpoint = "tcp://*:6002"

# setup the broker
broker = ublarcvserver.MDBroker(bindpoint, True)
broker.start()
print "broker started"

# setup the client
client = ublarcvserver.DummyClient(endpoint, True)
print "client created"

# setup the worker
worker = ublarcvserver.DummyWorker(endpoint, True)
print "worker created"

client.request()

print "[ENTER] to end"
raw_input()
