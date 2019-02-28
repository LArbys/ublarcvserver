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

  protected:

    void create(std::string server_addr);

    // job loop
    bool do_job();

    // function user provides to process message
    virtual MDWorkerMsg_t process_message(const int ninputframes,
                                          const int nresponses_to_frame,
                                          char* msg) = 0;

    //void signalHandler(int sig);
    void destroyWorker();

  private:

    std::string _service_name;
    std::string _id_name;
    std::string _server_addr;

    // the worker object
    mdp_worker_t* _pworker;

    // instance counter, used for naming
    static size_t _ninstances;
    static sig_atomic_t _signaled;

    // returns prefix for error messages
    std::string error_prefix() const;
  };

}

#endif
