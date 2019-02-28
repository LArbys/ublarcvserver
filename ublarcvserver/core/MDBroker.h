#ifndef __MDBROKER_H__
#define __MDBROKER_H__

#include <string>
#include "majordomo_library.h"

namespace ublarcvserver {

  class MDBroker {
  public:
    MDBroker( std::string broker_addr, bool verbose=false);
    virtual ~MDBroker();

    void start();

  protected:
    MDBroker() {};

  private:

    std::string _broker_addr;
    zactor_t *_pbroker;
    bool _verbose;

  };

}

#endif
