#ifndef __UBLCVS_MDWORKER_MSG_H__
#define __UBLCVS_MDWORKER_MSG_H__

namespace ublarcvserver {

  struct MDWorkerMsg_t {
    std::string str_msg; // the data to send as string
    zmsg_t*         msg; // data to send as zmsg
    int isfinal;         // if 1, then we are done responding, so send final
    MDWorkerMsg_t()
    : str_msg(""), msg(nullptr), isfinal(1)
     {};
  };


}

#endif
