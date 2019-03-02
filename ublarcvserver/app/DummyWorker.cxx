#include "DummyWorker.h"

#include <iostream>

namespace ublarcvserver {

  MDWorkerMsg_t DummyWorker::process_message(const int nresponses_to_frame,
                                             zmsg_t* msg) {
    MDWorkerMsg_t out;
    out.str_msg = "hello";
    out.isfinal = 1;
    std::cout << "[DummyWorker]: replying to client msg" << std::endl;
    std::cout << "worker msg: " << out.str_msg << std::endl;
    std::cout.flush();
    return out;
  }

}
