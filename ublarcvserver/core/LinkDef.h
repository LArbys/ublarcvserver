//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// Classes
#pragma link C++ namespace ublarcvserver+;
#pragma link C++ namespace ublarcvserver::zmqutil+;
//#pragma link C++ struct zframe_t;
#pragma link C++ struct ublarcvserver::zmqutil::zmsg_t;
#pragma link C++ struct ublarcvserver::MDWorkerMsg_t+;
#pragma link C++ class ublarcvserver::MDWorkerBase+;
#pragma link C++ class ublarcvserver::MDClientBase+;
//ADD_NEW_CLASS ... do not change this line

#endif
