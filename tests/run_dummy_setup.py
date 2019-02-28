import os,sys
from ublarcvserver import ublarcvserver

"""
This script is used to test the Majordomo classes.

We implement a dummy setup where the client and worker just say hello to each other.
Also servers as an example.

We also setup the basic larcv client and worker, which pass larcv images back and forth.
"""

# setup the broker
broker = ublarcvserver.MDBroker("tcp://*:6002", True)
broker.start()



print broker


