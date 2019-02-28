#include "DummyClient.h"

#include <sstream>
#include <iostream>
#include <exception>

namespace ublarcvserver {


  /**
  *  send a dummy Message
  *
  */
  zmsg_t* DummyClient::make_request_message() {

    zmsg_t *msg = zmsg_new();
    assert(msg);
    int res = zmsg_addstr(msg, "Message");
    assert(res == 0);
    return msg;
  }

  /**
  * process the reply from the dummy worker
  * we do not do anything, except print the message
  *
  * @pararm[in] message pointer to zmq message
  */
  bool DummyClient::process_reply( zmsg_t* message ) {

    std::cout << "[Dummy Client] Received message ----- " << std::endl;
    zmsg_print( message );
    std::cout << "------------------------------------- " << std::endl;
    std::cout.flush();

    return true;
  }

}
