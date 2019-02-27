#ifndef __MDWORKER_BASE_H__
#define __MDWORKER_BASE_H__

/**
* \class MDWorkerBase
* \brief Base class for MajorDomo worker
*
*/

#include <string>

#include "majordomo_library.h"

namespace ublcvserver {

  class MDWorkerBase {

  public:
    MDWorkerBase( std::string service_name, std::string server_addr,
                  std::string idname="" );
    virtual ~MDWorkerBase(){};

    std::string get_service_name() { return _service_name; };
    std::string get_id_name() { return _id_name; };

    // starts worker
    void run();

    virtual void process_message() = 0;

  protected:

    void connect(std::string server_addr);

  private:

    std::string _service_name;
    std::string _id_name;
    std::string _server_addr;

    // the worker object
    mdp_worker_t* _pworker;

    // instance counter, used for naming
    static size_t _ninstances;

    // returns prefix for error messages
    std::string error_prefix() const;
  };

}

#endif
