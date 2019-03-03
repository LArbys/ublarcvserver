#include "MDWorkerBase.h"
#include <iostream>
#include <sstream>
#include <exception>
#include <csignal>

namespace ublarcvserver {

  size_t MDWorkerBase::_ninstances = 0;
  sig_atomic_t MDWorkerBase::_signaled = 0;

  void signalHandler(int sig) {
    std::cout << "SIGNAL " << sig << "interrupt seen" << std::endl;
    //destroyWorker();
    MDWorkerBase::set_signal();
  }

  MDWorkerBase::MDWorkerBase( std::string service_name,
    std::string server_addr, std::string id_name, bool verbose )
  : _service_name(service_name),
    _server_addr(server_addr),
    _verbose(verbose),
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

     // define signal handler
     signal(SIGINT,  signalHandler);
     signal(SIGTERM, signalHandler);
     signal(SIGABRT, signalHandler);
    }

  /**
  * destructor
  *
  */
  MDWorkerBase::~MDWorkerBase() {
    destroyWorker();
  }

  /**
  *  destroy our connections to the broker
  *
  */
  void MDWorkerBase::destroyWorker() {
    if (!_pworker) return;
    //std::cout << "destroy mdp_worker" << std::endl;

    // free poller
    zpoller_destroy( &_worker_poll );
    _worker_poll = nullptr;
    _worker_sock = nullptr;

    mdp_worker_destroy(&_pworker);
    _pworker = nullptr;
  }

  /**
  * create mdp worker and connect to server
  */
  void MDWorkerBase::create(std::string server_addr ) {
    if ( _pworker ) {
      std::stringstream ss;
      ss << error_prefix(__FUNCTION__)
         << "asked to connect while still connected." << std::endl;
      throw std::runtime_error(ss.str());
    }
    _server_addr = server_addr;

    try {
      _pworker = mdp_worker_new( _server_addr.c_str(), _service_name.c_str() );
    }
    catch (std::exception& e ) {
      std::stringstream ss;
      ss << error_prefix(__FUNCTION__)
         << "could not create new mdp_worker_t: " << e.what()
         << std::endl;
      throw std::runtime_error(ss.str());
    }

    //  get socket
    _worker_sock = mdp_worker_msgpipe(_pworker);

    // make poller for socket
    _worker_poll = zpoller_new(_worker_sock);
  }

  /**
  * a string to prefix to error messages
  *
  * returns function name, file, line number
  */
  std::string MDWorkerBase::error_prefix(std::string func) const {
    std::stringstream ss;
    ss << "MDWorkerBase" << "::" << func << ":L" << __LINE__
         << ":ID[" << _id_name << "] :: ";
    return ss.str();
  }

  /**
  * start event loop. we poll the socket, waiting for messages.
  *
  */
  void MDWorkerBase::run() {
    //std::cout << "WORKER RUN" << std::endl;
    while (1) {
      bool job_performed = do_job();
      //if ( job_performed ) break;
      if ( _signaled )
        break;
    }
  }

  /**
  * perform one poll of the socket to the broker
  *
  * @param[in] timeout_secs time of timeout in seconds
  * @return true if socket found, false if timed out
  *
  */
  bool MDWorkerBase::pollSocket( float timeout_secs ) {
    int timeout_msecs = (int)(timeout_secs*1000);
    _pollin_socket = (zsock_t*)zpoller_wait(_worker_poll, timeout_msecs);


    if ( zpoller_expired(_worker_poll) ) {
      // poller ends due to time-out. exit this loop.
      if ( _verbose ) {
        std::cout << error_prefix(__FUNCTION__)
                  << "poller expired. "
                  << "timeout=" << timeout_msecs << "msecs" << std::endl;
      }
      return false;
    }
    if (_verbose ) {
      std::cout << error_prefix(__FUNCTION__)
                << "pollin socket found input: " << _pollin_socket
                << " timeout=" << timeout_msecs << std::endl;
    }

    return true;
  }

  bool MDWorkerBase::do_job() {

    // poll broker socket
    bool has_pollin = pollSocket(10.0);
    if ( !has_pollin )
      return false;

    // if got an input, keep going

    // get request from the broker
    if (_verbose )
      std::cout << error_prefix(__FUNCTION__)
                << "get broker command (blocking)" << std::endl;
    char* cmd = nullptr;
    try {
      cmd = zstr_recv(_worker_sock);
      //std::cout << "Got command from client: " << std::string(cmd) << std::endl;
    }
    catch (std::exception& e) {
      std::stringstream ss;
      ss << __FUNCTION__ << "::" << __FILE__ << "." << __LINE__
         << ": error tryting to get job: " << e.what() << std::endl;
      throw std::runtime_error(ss.str());
    }
    if ( _verbose )
      std::cout << error_prefix(__FUNCTION__) << "broker message: " << cmd << std::endl;

    // get the start of the message: expect "frame,message" format
    zframe_t *address; // frame
    zmsg_t *message;   // message body
    int res = zsock_recv(_worker_sock, "fm", &address, &message);

    // now process the message
    int npartial_out = 0; // number of messages out
    bool sent_final   = false;
    while ( !sent_final ) {
      // loop will stop once, frame_start is nullptr
      //char *frame_message = zframe_strdup(current_frame);

      bool done_w_frame = false;
      //zmsg_t* msg_response = zmsg_new();
      MDWorkerMsg_t response;

      // get response to message
      //std::cout << "call user process_message" << std::endl;
      response = process_message( npartial_out, message );
      //std::cout << "response: " << response.msg << std::endl;
      //std::cout.flush();

      // add reply to message
      if ( !response.msg ) {
        // msg not filled, so use string
        response.msg = zmsg_new();
        zmsg_addstr( response.msg, response.str_msg.c_str() );
      }
      //else {
      //  // msg filled, so use this
      //  std::cout << "add msg: "
      //            << " nframes=" << zmsg_size(response.msg)
      //            << " frame1=" << zframe_size(zmsg_first(response.msg))
      //            << std::endl;
      //  zmsg_addmsg(msg_response, &(response.msg) );
      //}
      //npartial_out++;

      // send the message
      if ( !response.isfinal ) {
        // not the final chunk, so we send a partial chunk
        // Make a copy of address, because mdp_worker_send_partial will free it
        //std::cout << "respond partial." << std::endl;
        zframe_t *address2 = zframe_dup(address);
        mdp_worker_send_partial(_pworker, &address2, &response.msg);
        npartial_out++;
        // go to next frame, or if final.
      }
      else {
        // final, we send final frame_message. send the last address
        //std::cout << "[MDWorkerBase] respond to frame with final msg."
        //          << " number of frames=" << zmsg_size(response.msg)
        //          << " frame1=" << zframe_size(zmsg_first(response.msg))
        //          << std::endl;
        mdp_worker_send_final( _pworker, &address, &response.msg );
        sent_final = true;
        npartial_out++;
      }

    } // end of reply loop


    return true;
  }//end of do_job

  /**
  * get broker command
  *
  */
  std::string MDWorkerBase::getBrokerCommand() {
    char* cmd = zstr_recv(_worker_sock);
    std::string scmd = cmd;
    free(cmd);
    return scmd;
  }

  MDWorkerMsg_t MDWorkerBase::process_message(
    const int nresponses_to_message, zmsg_t* input_msg)
  {
    std::stringstream ss;
    ss << error_prefix(__FUNCTION__)
       << ": should be overridden by child class"
       << std::endl;
    throw std::runtime_error(ss.str());
    MDWorkerMsg_t out;
    return out;
  }



}
