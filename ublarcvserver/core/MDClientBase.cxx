#include "MDClientBase.h"

#include <iostream>
#include <sstream>
#include <exception>
#include <string>

namespace ublarcvserver {

  MDClientBase::MDClientBase(std::string broker_addr,
    std::string service_name, bool verbose )
   : _broker_addr(broker_addr),
     _service_name(service_name),
     _client(nullptr)
  {
    _client = mdp_client_new(_broker_addr.c_str());

    if ( !_client ) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << "MDP Client could not be created." << std::endl;
      throw std::runtime_error(ss.str());
    }

    if ( verbose )
      mdp_client_set_verbose(_client);
  }

  MDClientBase::~MDClientBase() {
    close_mdp_client();
  }

  void MDClientBase::close_mdp_client() {
    try {
      mdp_client_destroy(&_client);
      _client = nullptr;
    }
    catch( std::exception& e ) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << "MDP Client could not be closed: " << e.what() << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  /**
  *  make a request of a worker (through the broker)
  *
  *  uses the concrete methods create_message and process_reply
  *   that the user must supply.
  */
  void MDClientBase::request() {

    zmsg_t *msg = make_request_message();
    if ( !msg ) {
      std::stringstream ss;
      ss << __FUNCTION__ << ":" << __FILE__ << "." << __LINE__ << " :: "
         << "error making message (returned nullptr)" << std::endl;
      throw std::runtime_error(ss.str());
    }

    // send the message, we send to workers providing the "dummy" service
    int res = mdp_client_request(_client, _service_name.c_str(), &msg);
    //std::cout << "[DummyClient] sent request" << std::endl;

    // get the client socket
    zsock_t *client_sock = mdp_client_msgpipe(_client);

    // make a poller in order to be able to get back control after
    // some fixed time
    zpoller_t* client_poll = zpoller_new(client_sock);

    bool hasfinal = false;
    int ntimeouts = 0;
    int maxtimeouts = 10;
    zmsg_t* full_reply = zmsg_new();

    while (!hasfinal || ntimeouts>=maxtimeouts) {
      // poll for input message
      int timeout_secs = (ntimeouts+1)*10;

      zsock_t* pollin_socket =
          (zsock_t*)zpoller_wait(client_poll, timeout_secs*1000);

      if ( zpoller_expired(client_poll) ) {
        ntimeouts++;
        if ( ntimeouts<maxtimeouts)
          continue; // try again
        else
          break; // end loop in failed state
      }

      //std::cout << "poller found input command" << std::endl;
      char* cmd   = 0;
      zmsg_t *frame_reply = nullptr;

      res = zsock_recv(client_sock, "sm", &cmd, &frame_reply);
      //std::cout << "[MDClientBase] Number of frames in message received: "
      //          << zmsg_size(frame_reply)
      //          << std::endl;

      if ( frame_reply ) {
        zframe_t *current_frame = zmsg_pop(frame_reply);
        while ( current_frame ) {
          zmsg_append( full_reply, &current_frame );
          current_frame = zmsg_pop(frame_reply);
        }
        //std::cout << "[MDClientBase] Append message."
        //          << " nframes=" << zmsg_size( full_reply )
        //          << std::endl;
      }

      //printf("Client (2): got command %s\n", cmd);
      //std::cout << " Response body: " << std::endl;
      //zmsg_print(message);
      if (std::string(cmd)=="PARTIAL") {
        //std::cout << "continue reading." << std::endl;
      }
      else {
        //std::cout << "[MDClientBase] received last frame." << std::endl;
        hasfinal = true; // will stop the loop
      }
      //zmsg_print(full_reply);
      //std::cout.flush();
    }//end of receiving reply loop

    // now that message has been collected, process it
    // this is a user function
    process_reply( full_reply );

    // we're done with the message, destroy it
    zmsg_destroy( &msg );
    zmsg_destroy( &full_reply );

    // done with the poller
    zpoller_destroy( &client_poll );
  }
}
