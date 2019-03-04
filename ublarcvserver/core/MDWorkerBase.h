#ifndef __MDWORKER_BASE_H__
#define __MDWORKER_BASE_H__

/**
* \class MDWorkerBase
* \brief Base class for MajorDomo worker
*
*/

#include <string>

#include "majordomo_library.h"
#include "MDWorkerMsg.h"

namespace ublarcvserver {

  class MDWorkerBase {

  public:
    MDWorkerBase( std::string service_name, std::string server_addr,
                  std::string idname="", bool verbose=false );
    virtual ~MDWorkerBase();

    std::string get_service_name() { return _service_name; };
    std::string get_id_name() { return _id_name; };

    // starts worker loop
    void run();
    static void set_signal() { _signaled=1; };
    // wait for message
    bool pollSocket( float timeout_secs );
    
  protected:

    void create(std::string server_addr);

    // job loop
    bool do_job();

    // function user provides to process message
    virtual MDWorkerMsg_t process_message(const int nresponses_to_message,
                                          zmsg_t* input_msg);

    //void signalHandler(int sig);
    void destroyWorker();

    // get broker command
    std::string getBrokerCommand();

    // get client message
    ///std::vector<std::string>


  private:

    std::string _service_name;
    std::string _id_name;
    std::string _server_addr;
    bool        _verbose;

    // the worker object
    mdp_worker_t* _pworker;

    // worker socket to broker
    zsock_t*      _worker_sock;

    // socket poller
    zpoller_t*    _worker_poll;

    // socket from poller
    zsock_t*      _pollin_socket;

    // instance counter, used for naming
    static size_t _ninstances;
    static sig_atomic_t _signaled;

    // returns prefix for error messages
    std::string error_prefix(std::string) const;
  };

}

#endif
