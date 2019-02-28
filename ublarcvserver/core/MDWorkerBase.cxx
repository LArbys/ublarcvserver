#include "MDWorkerBase.h"
#include <iostream>
#include <sstream>
#include <exception>

namespace ublarcvserver {

  size_t MDWorkerBase::_ninstances = 0;

  MDWorkerBase::MDWorkerBase( std::string service_name,
    std::string server_addr, std::string id_name, bool verbose )
  : _service_name(service_name),
    _server_addr(server_addr),
    _pworker(nullptr)
    {
      // create the id name
      std::stringstream ss;
      if (id_name!="")
        ss << _service_name << "_" << id_name << "_" << _ninstances;
      else
        ss << _service_name << "_" << _ninstances;
      _id_name = ss.str();
      _ninstances++;

      create(_server_addr);

      if ( verbose )
        mdp_worker_set_verbose(_pworker);
    }

  /**
  * destructor
  *
  */
  MDWorkerBase::~MDWorkerBase() {
    std::cout << "destroy mdp_worker" << std::endl;
    mdp_worker_destroy(&_pworker);
    _pworker = nullptr;
  }

  /**
  * create mdp worker and connect to server
  */
  void MDWorkerBase::create(std::string server_addr ) {
    if ( _pworker ) {
      std::stringstream ss;
      ss << error_prefix()
         << "asked to connect while still connected." << std::endl;
      throw std::runtime_error(ss.str());
    }
    _server_addr = server_addr;

    try {
      _pworker = mdp_worker_new( _server_addr.c_str(), _service_name.c_str() );
    }
    catch (std::exception& e ) {
      std::stringstream ss;
      ss << error_prefix()
         << "could not create new mdp_worker_t: " << e.what()
         << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  /**
  * a string to prefix to error messages
  *
  * returns function name, file, line number
  */
  std::string MDWorkerBase::error_prefix() const {
    std::stringstream ss;
    ss << __FUNCTION__ << ":" << __FILE__ << ":" << __LINE__
         << "MDWorker[" << _id_name << "] :: ";
    return ss.str();
  }

  /**
  * start event loop. we poll the socket, waiting for messages.
  *
  */
  void MDWorkerBase::run() {
    std::cout << "WORKER RUN" << std::endl;
    /*
    while (1) {
      do_job();
    }
    */
    do_job();
  }

  void MDWorkerBase::do_job() {
    // get socket for worker
    zsock_t *worker_sock = mdp_worker_msgpipe(_pworker);

    // get request from the broker
    char* cmd = nullptr;
    try {
      cmd = zstr_recv(worker_sock);
      //std::cout << "Got command from client: " << std::string(cmd) << std::endl;
    }
    catch (std::exception& e) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << ": error tryting to get job: " << e.what() << std::endl;
      throw std::runtime_error(ss.str());
    }

    // get the start of the message: expect "frame,message" format
    zframe_t *address; // frame
    zmsg_t *message;   // message body
    int res = zsock_recv(worker_sock, "fm", &address, &message);

    // now process different parts with partial responses
    zframe_t *current_frame = zmsg_first(message);
    int nframes_in   = 0; // number of frames we've read
    int npartial_out = 0; // number of messages out
    bool sent_final   = false;
    while ( current_frame ) {
      // loop will stop once, frame_start is nullptr
      char *frame_message = zframe_strdup(current_frame);

      bool done_w_frame = false;
      zmsg_t* msg_response = zmsg_new();
      int nresponses_to_frame = 0;
      MDWorkerMsg_t response;
      while ( !done_w_frame ) {
        // get response to message
        //std::cout << "call user process_message" << std::endl;
        response = process_message( nframes_in,
                                    nresponses_to_frame,
                                    frame_message );
        //std::cout << "response: " << response.msg << std::endl;
        std::cout.flush();
        zmsg_addstr(msg_response, response.msg.c_str() );
        nresponses_to_frame++;
        if ( response.done_with_frame==1 ) {
          done_w_frame = true;
          break;
        }
      }

      // send the message
      if ( !response.isfinal ) {
        // not the final chunk, so we send a particle chunk
        // Make a copy of address, because mdp_worker_send_partial will free it
        //std::cout << "respond partial." << std::endl;
        zframe_t *address2 = zframe_dup(address);
        mdp_worker_send_partial(_pworker, &address2, &msg_response);
        npartial_out++;
        // go to next frame, or if final.
        current_frame = zmsg_next(message);
      }
      else {
        // final, we send final frame_message. spend the last address
        //std::cout << "respond to frame with final msg" << std::endl;
        mdp_worker_send_final( _pworker, &address, &msg_response );
        sent_final = true;
        npartial_out++;
      }
      nframes_in++;
      if ( response.isfinal || !current_frame )
        break;
    } // end of frame while loop

    if ( !sent_final ) {
      zmsg_t* msg_response = zmsg_new();
      char finmsg[50];
      sprintf( finmsg, "Final auto-response" );
      zmsg_addstr(msg_response, finmsg);
      mdp_worker_send_final( _pworker, &address, &msg_response );
    }

  }//end of do_job
}
