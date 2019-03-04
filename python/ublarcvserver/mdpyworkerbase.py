import zmq
import time, abc, logging
import ROOT
import protocol as p
from ROOT import ublarcvserver

class MDPyWorkerBase(object):
    __metaclass__ = abc.ABCMeta
    _ninstances = 0


    def __init__(self, service_name, broker_address,
                    zmq_context=None, id_name=None, verbose=False,
                    heartbeat_interval_secs=2.5,
                    heartbeat_timeout_secs=10.0 ):
        self._broker_address = broker_address
        self._service_name   = service_name.encode('ascii')
        self._verbose        = verbose
        self._heartbeat_timeout  = heartbeat_timeout_secs
        self._heartbeat_interval = heartbeat_interval_secs

        # create the socket we need
        # its a simple request/reply socket
        print "zmq_context:",zmq_context
        self._context = zmq_context if zmq_context else zmq.Context()

        self._socket = None  # type: zmq.Socket
        self._poller = None  # type: zmq.Poller
        #self._linger = zmq_linger
        self._last_broker_hb = 0.0
        self._last_sent_message = 0.0

        if not id_name:
            self.id_name = "{}:{}".format(self._service_name,
                                          MDPyWorkerBase._ninstances)
        MDPyWorkerBase._ninstances += 1

        self._log = logging.getLogger(__name__)

    # ABSTRACT METHODS
    # -------------------------------------------------------------
    @abc.abstractmethod
    def make_reply(self,request):
        """ user should implement this. recieve multipart message and return
            multipart message"""
        raise RuntimeError("should not get here")

    # PUBLIC METHODS
    # ------------------------------------------------------------

    def is_connected(self):
        return self._socket is not None

    def connect(self, reconnect=False):
        # type: (bool) -> None
        if self.is_connected():
            if not reconnect:
                return
            self._disconnect()

        # Set up socket
        self._socket = self._context.socket(zmq.DEALER)
        #self._socket.setsockopt(zmq.LINGER)
        self._socket.connect(self._broker_address)
        self._log.debug("Connected to broker on ZMQ DEALER socket at %s",
                        self._broker_address)

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        self._send_ready()
        self._last_broker_hb = time.time()

    def disconnect(self, reconnect=False):
        if self.is_connected():
            return
        self._send_disconnect()
        self._disconnect()

    def idname(self):
        return self.id_name

    def wait_for_request(self):
        # type: () -> Tuple[bytes, List[bytes]]
        """Wait for a REQUEST command from the broker and return the
        client address and message body frames.
        Will internally handle timeouts, heartbeats and check for
        protocol errors and disconnect commands.
        """
        command, frames = self._receive()

        if command == p.DISCONNECT:
            self._log.debug("Got DISCONNECT from broker; Disconnecting")
            self._disconnect()
            #raise RuntimeError("Disconnected on message from broker")
            return None, None

        elif command != p.REQUEST:
            raise RuntimeError("Unexpected message type from broker: {}"
                                .format(command))

        if len(frames) < 3:
            raise RuntimeError("Unexpected REQUEST message size, got {} frames,\
                                expecting at least 3".format(len(frames)))

        client_addr = frames[0]
        request = frames[2:]
        return client_addr, request

    def send_reply_final(self, client, frames):
        # type: (bytes, List[bytes]) -> None
        """Send final reply to client
        FINAL reply means the client will not expect any additional parts to the reply. This should be used
        when the entire reply is ready to be delivered.
        """
        self._send_to_client(client, p.FINAL, *frames)

    def send_reply_partial(self, client, frames):
        # type: (bytes, List[bytes]) -> None
        """Send the given set of frames as a partial reply to client
        PARTIAL reply means the client will expect zero or more additional PARTIAL reply messages following
        this one, with exactly one terminating FINAL reply following. This should be used if parts of the
        reply are ready to be sent, and the client is capable of processing them while the worker is still
        at work on the rest of the reply.
        """
        self._send_to_client(client, p.PARTIAL, *frames)

    def run(self):
        """ loops until we get a disconnect signal from the broker. Or
            until an error occurs"""
        while True:
            client_addr, request = self.wait_for_request()
            if client_addr is None:
                break

            isfinal = False
            while not isfinal:
                # use the user function
                reply, isfinal = self.make_reply(request)
                if not isfinal:
                    self.send_reply_partial(client_addr,reply)
                else:
                    self.send_reply_final(client_addr,reply)



    # INTERNAL UTILS
    # -----------------------------------------------------------
    def _disconnect(self):
        if not self.is_connected():
            return
        self._socket.disconnect(self._broker_address)
        self._socket.close()
        self._socket = None
        self._last_sent_message -= self._heartbeat_interval

    def _send(self, message_type, *args):
        # type: (bytes, *bytes) -> None
        self._socket.send_multipart((b'',p.WORKER_HEADER, message_type) + args)
        self._last_sent_message = time.time()

    def _send_ready(self):
        self._send(p.READY,self._service_name)

    def _send_disconnect(self):
        self._send(p.DISCONNECT)

    def _send_to_client(self, client, message_type, *frames):
        self._send(message_type, client, b'', *frames)

    def _check_send_heartbeat(self):
        if time.time()-self._last_sent_message >= self._heartbeat_interval:
            self._log.debug("Sending HEARTBEAT to broker")
            self._send(p.HEARTBEAT)

    def _get_poll_timeout(self):
        # type: () -> int
        """Return the poll timeout for the current iteration in milliseconds
        """
        interval = int((time.time() - \
                        self._last_sent_message+self._heartbeat_interval) * 1000)
        return max(0,interval)

    @staticmethod
    def _verify_message(message):
        # type: (List[bytes]) -> Tuple[bytes, List[bytes]]
        if len(message) < 3:
            raise RuntimeError("Unexpected message length, \
                                expecting at least 3 frames, got {}"
                                .format(len(message)))

        if message.pop(0) != b'':
            raise RuntimeError("Expecting first message frame to be empty")

        if message[0] != p.WORKER_HEADER:
            print(message)
            raise RuntimeError("Unexpected protocol header [{}], expecting [{}]"
                                .format(message[0], p.WORKER_HEADER))

        if message[1] not in {p.DISCONNECT, p.HEARTBEAT, p.REQUEST}:
            raise RuntimeError("Unexpected message type [{}], \
                                expecting either HEARTBEAT, REQUEST or "
                                "DISCONNECT".format(message[1]))

        return message[1], message[2:]

    # INTERVAL MAIN LOOP
    # -------------------------------------------------------------------
    def _receive(self):
        """Poll on the socket until a command from the broker is received"""
        while True:
            # sanity check
            if self._socket is None:
                raise RuntimeError("Worker is disconnected")

            # time until we need to send a heartbeat
            self._check_send_heartbeat()
            poll_timeout = self._get_poll_timeout()

            try:
                socks = dict(self._poller.poll(timeout=poll_timeout))
            except zmq.error.ZMQError:
                # Probably connection was explicitly closed
                if self._socket is None:
                    continue
                raise

            if socks.get(self._socket) == zmq.POLLIN:
                message = self._socket.recv_multipart()
                self._log.debug("Got message of %d frames", len(message))
            else:
                self._log.debug("Receive timed out after %d ms", poll_timeout)
                if (time.time()-self._last_broker_hb)>self._heartbeat_timeout:
                    # We're not connected anymore?
                    self._log.info("Got no heartbeat in %d sec, \
                                    disconnecting and reconnecting socket",
                                    self._heartbeat_timeout)
                    self.connect(reconnect=True)
                continue

            command, frames = self._verify_message(message)
            self._last_broker_hb = time.time()

            if command == p.HEARTBEAT:
                self._log.debug("Got heartbeat message from broker")
                continue

            return command, frames
