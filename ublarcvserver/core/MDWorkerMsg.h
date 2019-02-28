#ifndef __UBLCVS_MDWORKER_MSG_H__
#define __UBLCVS_MDWORKER_MSG_H__

namespace ublarcvserver {

  struct MDWorkerMsg_t {
    std::string msg;     // the data to send
    int done_with_frame; // if 1, then we are done with frame
    int isfinal;         // if 1, then we are done responding, so send final
    MDWorkerMsg_t()
    : msg(""), done_with_frame(1), isfinal(0) {};
  };


}

#endif
