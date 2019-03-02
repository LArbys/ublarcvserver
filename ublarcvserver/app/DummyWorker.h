#ifndef __UBLCVS_DUMMY_WORKER_H__
#define __UBLCVS_DUMMY_WORKER_H__

#include "ublarcvserver/core/MDWorkerBase.h"
#include "ublarcvserver/core/MDWorkerMsg.h"

namespace ublarcvserver {

  class DummyWorker : public MDWorkerBase {
  public:

    DummyWorker(std::string broker_addr, bool verbose)
    : MDWorkerBase("dummy",broker_addr,"",verbose)
    {};

    virtual ~DummyWorker() {};

  protected:

    MDWorkerMsg_t process_message(const int nresponses_to_frame,
                                  zmsg_t* msg);

  };


}


#endif
