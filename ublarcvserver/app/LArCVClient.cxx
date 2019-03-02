#include "LArCVClient.h"

#include "larcv/core/json/json_utils.h"

namespace ublarcvserver {

  /**
  * add an image to send. represent it as a dense matrix
  *
  *  @param[in] img Image2D to store
  */
  void LArCVClient::addImage( const larcv::Image2D& img ) {
    //std::cout << __FUNCTION__ << std::endl;
    _images_toworker_v.push_back( ImgStoreMeta_t(&img, 0, 0, true ) );
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
    //std::cout << __FUNCTION__ << std::endl;
    _images_toworker_v.push_back( ImgStoreMeta_t(&img,0,threshold,false));
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
    //std::cout << __FUNCTION__ << std::endl;
    _images_toworker_v.push_back(
      ImgStoreMeta_t(&img,&img_select,threshold,false)
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
        if ( !imgstore.select ) {
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
    //std::cout << "LArCVClient: make request_message. "
    //          << "number of frames=" << zmsg_size(msg) << std::endl;
    return msg;
  }

  /**
  * turn zmq image into list of image2Ds
  */
  bool LArCVClient::process_reply( zmsg_t* msg ) {
    // clear out past images
    // also, since we received a reply, we already sent messages, so
    //   clear out input meta data
    _images_toworker_v.clear();
    _images_received_v.clear();

    //std::cout << "larcvclient: process_reply. number of messages= "
    //          << zmsg_size(msg)
    //          << std::endl;

    zframe_t* img_frame = zmsg_first(msg);
    while ( img_frame ) {
      //std::cout << "LArCVClient: start message parsing" << std::endl;


      // frame: image type
      char* imgtype = zframe_strdup(img_frame);
      std::string strimgtype = imgtype;
      free(imgtype);
      //std::cout << "image type: " << strimgtype << std::endl;
      img_frame = zmsg_next(msg);

      // frame: data
      size_t nbytes = zframe_size(img_frame);
      std::vector<uint8_t> bson( nbytes );
      memcpy( bson.data(), zframe_data(img_frame), nbytes*sizeof(uint8_t) );
      //std::cout << strimgtype << " msg size: " << nbytes << std::endl;

      if ( strimgtype=="denseimg2d" ) {
        //std::cout << "dense image2d from bson" << std::endl;
        larcv::Image2D img2d = larcv::json::image2d_from_bson( bson );
        _images_received_v.emplace_back( std::move(img2d) );
      }
      else if ( strimgtype=="sparseimg2d") {
        //std::cout << "sparse image2d from bson" << std::endl;
        larcv::Image2D img2d=larcv::json::image2d_from_bson_pixelarray(bson);
        _images_received_v.emplace_back( std::move(img2d) );
      }
      else {
        return false;
      }

      // next frame
      //std::cout << "get next frame" << std::endl;
      img_frame = zmsg_next(msg);
    }
    return true;
  }

  /**
  * take the images from the client
  *
  * @param[inout] image_v Vector for image2D
  *
  */
  void LArCVClient::takeImages( std::vector<larcv::Image2D>& image_v ) {
    for ( auto &img : _images_received_v ) {
      image_v.emplace_back( std::move(img) );
    }
    _images_received_v.clear();
  }

}
