#include "MirrorWorker.h"

#include <iostream>

namespace ublarcvserver {

  MDWorkerMsg_t MirrorWorker::process_message(const int nresponses_to_frame,
                                              zmsg_t* msg) {
    MDWorkerMsg_t out;
    out.msg = zmsg_dup(msg);
    out.isfinal = 1;
    std::cout << "[MirrorWorker]:"
              << " duplicating message and sending back." << std::endl;
    std::cout.flush();
    return out;
  }

}
