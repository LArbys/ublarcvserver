#include "MDWorkerBase.h"
#include <sstream>
#include <exception>

namespace ublarcvserver {

  size_t MDWorkerBase::_ninstances = 0;

  MDWorkerBase::MDWorkerBase( std::string service_name,
    std::string server_addr, std::string id_name )
  : _service_name(service_name),
    _server_addr(server_addr),
    _pworker(nullptr)
    {
      // create the id name
      std::stringstream ss;
      if (id_name!="")
        ss << _service_name << "_" << id_name << "_" << _ninstances;
      else
        ss << _service_name << "_" << _ninstances;
      _id_name = ss.str();
      _ninstances++;

      connect(_server_addr);
    }


  /**
  * create mdp worker and connect to server
  */
  void MDWorkerBase::connect(std::string server_addr ) {
    if ( _pworker ) {
      std::stringstream ss;
      ss << error_prefix()
         << "asked to connect while still connected." << std::endl;
      throw std::runtime_error(ss.str());
    }
    _server_addr = server_addr;

    try {
      _pworker = mdp_worker_new( _server_addr.c_str(), _service_name.c_str() );
    }
    catch (std::exception& e ) {
      std::stringstream ss;
      ss << error_prefix()
         << "could not create new mdp_worker_t: " << e.what()
         << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  /**
  * a string to prefix to error messages
  *
  * returns function name, file, line number
  */
  std::string MDWorkerBase::error_prefix() const {
    std::stringstream ss;
    ss << __FUNCTION__ << ":" << __FILE__ << ":" << __LINE__
         << "MDWorker[" << _id_name << "] :: ";
    return ss.str();
  }


}
