#ifndef __UBLCVS_LARCV_CLIENT_H__
#define __UBLCVS_LARCV_CLIENT_H__

/**
* \class LArCVClient
*
* This class provides a fairly generic way to load larcv images and pass
*  then to a remote worker.
*
* This assumes that the event loop is handled elsewhere.  Instead, one provides
*  the images to be shipped. One can then get the images from the worker back.
* The serialization and message handling is taken care of by this classes
*
*/

#include <vector>

#include "ublarcvserver/core/MDClientBase.h"
#include "larcv/core/DataFormat/Image2D.h"


namespace ublarcvserver {

  class LArCVClient : public MDClientBase {
  public:

    typedef enum {kDENSE=0,kSPARSE} ImageType_t;

    LArCVClient( std::string broker_addr,
      std::string service_name, bool verbose, ImageType_t imgtype=kDENSE )
    : MDClientBase(broker_addr, service_name, verbose ),
      _imgtype(imgtype)
    {};

    void addImage( const larcv::Image2D& );
    void addImageAsPixelList( const larcv::Image2D& img_value,
      const float threshold );
    void addImageAsPixelListWithSelection( const larcv::Image2D& img_value,
      const larcv::Image2D& img_select, const float threshold );
    void takeImages( std::vector<larcv::Image2D>& );

    // call request to send images to worker

  protected:

    // user provided concrete methods
    zmsg_t* make_request_message();
    bool process_reply( zmsg_t* );

    ImageType_t _imgtype;
    struct ImgStoreMeta_t {
      const larcv::Image2D* img;
      const larcv::Image2D* select;
      float threshold;
      bool isdense;
      ImgStoreMeta_t( const larcv::Image2D* fimg,
        const larcv::Image2D* fselect,
        float fthreshold, bool fisdense )
        : img(fimg), select(fselect), threshold(fthreshold), isdense(fisdense)
        {};
    };
    std::vector< ImgStoreMeta_t > _images_toworker_v;
    std::vector< larcv::Image2D > _images_received_v;

  };


}

#endif
