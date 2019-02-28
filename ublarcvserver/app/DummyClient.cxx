#include "DummyClient.h"

#include <sstream>
#include <iostream>
#include <exception>

namespace ublarcvserver {

  DummyClient::DummyClient( std::string broker_addr, bool verbose )
   : _broker_addr(broker_addr),
     _client(nullptr)
  {
    _client = mdp_client_new(_broker_addr.c_str());

    if ( !_client ) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << "DummyClient could not be started." << std::endl;
      throw std::runtime_error(ss.str());
    }

    if ( verbose )
      mdp_client_set_verbose(_client);
  }

  DummyClient::~DummyClient() {
    try {
      mdp_client_destroy(&_client);
      _client = nullptr;
    }
    catch( std::exception& e ) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << "DummyClient could not be closed: " << e.what() << std::endl;
      throw std::runtime_error(ss.str());
    }

  }

  /**
  *  send a dummy Message
  *
  */
  void DummyClient::request() {

    zmsg_t *msg = zmsg_new();
    assert(msg);
    int res = zmsg_addstr(msg, "Message");
    assert(res == 0);

    // send the message, we send to workers providing the "dummy" service
    std::string service = "dummy";
    res = mdp_client_request(_client, service.c_str(), &msg);
    std::cout << "[DummyClient] sent request" << std::endl;

    // get the client socket
    zsock_t *client_sock = mdp_client_msgpipe(_client);

    // blocking recv
    bool hasfinal = false;
    while (!hasfinal) {
      char* cmd;
      zmsg_t *message;
      res = zsock_recv(client_sock, "sm", &cmd, &message);
      printf("Client (2): got command %s\n", cmd);
      std::cout << " Response body: " << std::endl;
      zmsg_print(message);
      if (std::string(cmd)=="PARTIAL") {
        std::cout << "continue reading." << std::endl;
      }
      else {
        std::cout << "[DummyClient] received last frame." << std::endl;
        hasfinal = true;
      }
      std::cout.flush();
      zmsg_destroy(&message);
    }
  }

}
