#ifndef __UBLCVS_MIRROR_WORKER_H__
#define __UBLCVS_MIRROR_WORKER_H__

#include "ublarcvserver/core/MDWorkerBase.h"
#include "ublarcvserver/core/MDWorkerMsg.h"

/**
*  \class MirrorWorker
*  \brief A worker that simply reflects the client message back.
*         useful for testing client messages
*
*/

namespace ublarcvserver {

  class MirrorWorker : public MDWorkerBase {
  public:

    MirrorWorker(std::string broker_addr, bool verbose)
    : MDWorkerBase("mirror",broker_addr,"",verbose)
    {};

    virtual ~MirrorWorker() {};

  protected:

    MDWorkerMsg_t process_message(const int nresponses_to_frame,
                                  zmsg_t* input_msg);

  };


}


#endif
