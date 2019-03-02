#include "LArCVClient.h"

#include "larcv/core/json/json_utils.h"

namespace ublarcvserver {

  /**
  * add an image to send. represent it as a dense matrix
  *
  *  @param[in] img Image2D to store
  */
  void LArCVClient::addImage( const larcv::Image2D& img ) {
    _images_toworker_v.push_back( ImgStoreMeta_t(&img, 0, 0, false ) );
  }

  /**
  * add an image to send. represent it as a sparse matrix
  *
  *  @param[in] img Image2D to store
  *  @param[in] threshold to use to select pixels
  */
  void LArCVClient::addImageAsPixelList(
    const larcv::Image2D& img,
    const float threshold )
  {
     _images_toworker_v.push_back( ImgStoreMeta_t(&img,0,threshold,true));
  }

  /**
  * add an image to send. represent it as a sparse matrix. use another image
  *   to select pixels.
  *
  *  @param[in] img Image2D to store
  *  @param[in] select Image2D to use to select pixels
  *  @param[in] threshold to use to select pixels
  */
  void LArCVClient::addImageAsPixelListWithSelection(
    const larcv::Image2D& img, const larcv::Image2D& img_select,
    const float threshold )
  {
     _images_toworker_v.push_back(
       ImgStoreMeta_t(&img,&img_select,threshold,true)
     );
  }


  /**
  * serialize the store of images into a message for zmq to pass
  *
  * uses images from _images_received_v
  *
  * @return a pointer to a msg_t object. it is assumed that we
  *  do not own the object
  *
  */
  zmsg_t* LArCVClient::make_request_message() {

    // we take the images from _images_received_v and serialize it into a
    // multiframe message

    // the message format is
    // [frame: bson for image]
    // [frame: bson for image]
    // ...
    zmsg_t* msg  = zmsg_new();

    for ( auto const& imgstore : _images_toworker_v ) {
      std::vector<uint8_t> bson;
      if ( imgstore.isdense ) {
        zmsg_addstr(msg,"denseimg2d");
        bson = larcv::json::as_bson( *(imgstore.img) );
      }
      else {
        zmsg_addstr(msg,"sparseimg2d");
        if ( imgstore.select ) {
          bson = larcv::json::as_bson_pixelarray( *imgstore.img,
                                                  imgstore.threshold );
        }
        else {
          bson = larcv::json::as_bson_pixelarray_withselection(*imgstore.img,
                    *imgstore.select, imgstore.threshold );
        }
      }

      zmsg_addmem( msg, (const void*)bson.data(), bson.size() );
    }
    return msg;
  }

  /**
  * turn zmq image into list of image2Ds
  */
  bool LArCVClient::process_reply( zmsg_t* msg ) {
    _images_received_v.clear();
    zframe_t* img_frame = zmsg_first(msg);
    while ( img_frame ) {
      // frame: image type
      char* imgtype = zframe_strdup(img_frame);
      img_frame = zmsg_next(msg);
      std::string strimgtype = imgtype;
      free(imgtype);

      // frame: data
      size_t nbytes = zframe_size(img_frame);
      std::vector<uint8_t> bson( nbytes );
      memcpy( bson.data(), zframe_data(img_frame), nbytes );


      if ( strimgtype=="denseimg2d" ) {
        larcv::Image2D img2d = larcv::json::image2d_from_bson( bson );
        _images_received_v.emplace_back( std::move(img2d) );
      }
      else if ( strimgtype=="sparseimg2d") {
        larcv::Image2D img2d=larcv::json::image2d_from_bson_pixelarray(bson);
        _images_received_v.emplace_back( std::move(img2d) );
      }
      else {
        return false;
      }

      // next frame
      img_frame = zmsg_next(msg);
    }
    return true;
  }

}
