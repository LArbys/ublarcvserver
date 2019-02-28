#include "MDBroker.h"

namespace ublarcvserver {

  MDBroker::MDBroker( std::string broker_addr, bool verbose )
  : _broker_addr(broker_addr),
    _pbroker(nullptr)
  {
    _pbroker = zactor_new(mdp_broker, (void*)"server");
    if ( verbose )
      zstr_send(_pbroker, "VERBOSE");
  }

  MDBroker::~MDBroker() {
    zactor_destroy(&_pbroker);
    _pbroker = nullptr;
  }

  void MDBroker::start() {
    zstr_sendx(_pbroker, "BIND", _broker_addr.c_str(), NULL);
  }



}
