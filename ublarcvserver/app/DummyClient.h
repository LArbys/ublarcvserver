#ifndef __UBLCVS_DUMMY_CLIENT_H__
#define __UBLCVS_DUMMY_CLIENT_H__

#include <string>

#include "majordomo_library.h"

namespace ublarcvserver {

  class DummyClient {
  public:
    DummyClient(std::string broker_addr, bool verbose);
    virtual ~DummyClient();

    void request();
    
  protected:


    std::string _broker_addr;
    mdp_client_t *_client;

  };

}

#endif
