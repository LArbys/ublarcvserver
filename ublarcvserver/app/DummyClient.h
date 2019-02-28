#ifndef __UBLCVS_DUMMY_CLIENT_H__
#define __UBLCVS_DUMMY_CLIENT_H__

#include <string>

#include "majordomo_library.h"
#include "ublarcvserver/core/MDClientBase.h"

namespace ublarcvserver {

  class DummyClient : public MDClientBase {

  public:

    DummyClient(std::string broker_addr, bool verbose)
    : MDClientBase( broker_addr, "dummy", verbose )
    {};

    virtual ~DummyClient() {};

  protected:

    // user provided concrete methods
    zmsg_t* make_request_message();
    bool process_reply( zmsg_t* );

  };

}

#endif
