import os,sys,time
from ublarcvserver import ublarcvserver
from multiprocessing import Process

"""
This script is used to test the Majordomo classes.

We implement a dummy setup where the client and worker just say hello to each other.
Also servers as an example.

We also setup the basic larcv client and worker, which pass larcv images back and forth.
"""
verbose = False

def start_worker( endpoint ):
    global verbose
    print "start worker on ",endpoint
    worker = ublarcvserver.DummyWorker(endpoint, verbose)
    print "worker started: ",worker.get_id_name()    
    worker.run()
    #while 1:
    #    time.sleep(1)

# endpoint:
endpoint  = "tcp://localhost:6005"
bindpoint = "tcp://*:6005"


# setup the worker
pworker = Process(target=start_worker,args=(endpoint,))
pworker.daemon = True
pworker.start()
#worker = ublarcvserver.DummyWorker(endpoint, True)
print "worker process created"

# setup the broker
broker = ublarcvserver.MDBroker(bindpoint, verbose)
broker.start()
print "broker started"

# setup the client
client = ublarcvserver.DummyClient(endpoint, verbose)
print "client created"

for x in xrange(5):
    print "REQUEST %d"%(x+1)
    client.request()
    time.sleep(2)

print "[ENTER] to end"
raw_input()
