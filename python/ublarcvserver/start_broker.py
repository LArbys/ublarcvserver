import os,sys
from multiprocessing import Process
from majortomo.broker import Broker

def start_broker(broker_bind_address):

    def start_broker_process(bindpoint):
        broker = Broker(bind=bindpoint)
        broker.run()

    pbroker = Process(target=start_broker_process,
                      args=(broker_bind_address,))
    pbroker.daemon = True
    pbroker.start()

    return pbroker
