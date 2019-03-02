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
#pragma link C++ class ublarcvserver::MDBroker+;
#pragma link C++ class ublarcvserver::DummyClient+;
#pragma link C++ class ublarcvserver::DummyWorker+;
#pragma link C++ struct ublarcvserver::LArCVClient::ImgStoreMeta_t;
#pragma link C++ class ublarcvserver::LArCVClient+;
#pragma link C++ class ublarcvserver::MirrorWorker+;
//ADD_NEW_CLASS ... do not change this line

#endif
