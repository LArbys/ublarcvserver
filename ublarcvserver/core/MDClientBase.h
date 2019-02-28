#ifndef __UBLCVS_MDCLIENT_BASE_H__
#define __UBLCVS_MDCLIENT_BASE_H__

#include "majordomo_library.h"
#include <string>

namespace ublarcvserver {

  class MDClientBase {
  public:

    MDClientBase(std::string broker_addr,
      std::string service_name, bool verbose);
    virtual ~MDClientBase();

    void request();

  protected:

    // routines user must define in her concrete implementation

    // produce the message to send to a worker via the broker
    virtual zmsg_t* make_request_message()     = 0;

    // process the reply from the worker
    // it is assumed that the base class still owns the zmsg_t instance
    virtual bool process_reply( zmsg_t* ) = 0;

  private:

    void close_mdp_client();

    std::string _broker_addr;
    std::string _service_name;
    mdp_client_t *_client;

  };

}

#endif
