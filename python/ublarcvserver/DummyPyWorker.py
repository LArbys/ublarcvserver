import os,sys
from .mdpyworkerbase import MDPyWorkerBase

class DummyPyWorker(MDPyWorkerBase):

    def __init__(self,broker_addr,**kwargs):
        super(DummyPyWorker,self).__init__("dummy",broker_addr,**kwargs)

    def make_reply(self,request,nreplies):
        """we simply send the info back"""
        #print("DummyPyWorker. Sending client message back")
        request = ["from {}".format(self.idname())]+request
        isfinal = True
        return request,isfinal
