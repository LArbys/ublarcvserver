import os,sys,time
import logging
from ublarcvserver import ublarcvserver
from ublarcvserver import DummyPyWorker, Broker, Client
from multiprocessing import Process

"""
This script is used to test the Majordomo classes.

We implement a dummy setup where the client and worker just say hello to each other.
Also servers as an example.

We also setup the basic larcv client and worker, which pass larcv images back and forth.
"""
verbose = True

def start_worker( endpoint, worker_verbose ):
    print "start worker on ",endpoint
    worker = DummyPyWorker(endpoint,verbose=True)
    worker.connect()
    print "worker started: ",worker.idname()
    #time.sleep(5)
    worker.run()
    #while True:
    #time.sleep(1)

    print "ending start-worker process"

def start_broker(bindpoint):
    print "start broker"
    broker = Broker(bind=bindpoint)
    broker.run()
    print "broker closed"

# endpoint:
endpoint  = "tcp://localhost:6005"
bindpoint = "tcp://*:6005"


logging.basicConfig(level=logging.DEBUG)

# setup the broker
#broker = ublarcvserver.MDBroker(bindpoint, verbose)
#broker.start()
pbroker = Process(target=start_broker,args=(bindpoint,))
pbroker.daemon = True
pbroker.start()

# setup the worker
pworker = Process(target=start_worker,args=(endpoint,verbose))
pworker.daemon = True
pworker.start()
print "worker process created"

# setup the client
client = Client(endpoint)
client.connect()
print "client connected"

for x in xrange(5):
    print "REQUEST %d"%(x+1)
    client.send("dummy","hello world %d"%(x))
    msg = client.recv_all_as_list()
    print "reply from worker: ",msg
    time.sleep(2)

print "[ENTER] to end"
raw_input()
