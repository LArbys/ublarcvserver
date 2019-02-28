#include "DummyWorker.h"

#include <iostream>

namespace ublarcvserver {

  MDWorkerMsg_t DummyWorker::process_message(const int ninputframes,
                                             const int nresponses_to_frame,
                                             char* msg) {
    MDWorkerMsg_t out;
    out.msg = "Partial response to "+std::string(msg);
    out.done_with_frame = 1;
    out.isfinal = 0;
    std::cout << "[DummyWorker]: replying to client msg" << std::endl;
    std::cout << "worker msg: " << out.msg << std::endl;
    std::cout.flush();
    return out;
  }

}
