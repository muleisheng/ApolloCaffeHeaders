#ifndef DETECT2D_LAYERS_BLOCKPACK_HPP_
#define DETECT2D_LAYERS_BLOCKPACK_HPP_

#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detect2d/NewRectMap.hpp"
#include "caffe/util/util_img.hpp"
namespace caffe {
/* Packing and unpacking image blocks. */
template<typename Dtype>
class BlockPack {
public:
    BlockPack();
    virtual ~BlockPack();
    virtual void SetUpParameter(
            const BlockPackingParameter& block_packing_param);

    virtual void imgpack_cpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out);
    virtual void imgpack_gpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out);

    int serial_roirect_blob(Blob<Dtype>& blob, NewRoiRect<Dtype>& roi_rect,
            int start);
    int read_roirect_serial(Blob<Dtype>& blob, NewRoiRect<Dtype>& roi_rect,
            int start);

    int serial_rect_blob(Blob<Dtype>& blob, NewRect& rect, int start);
    int read_rect_serial(Blob<Dtype>& blob, NewRect& rect, int start);

    virtual int serial_blockpack_blob(Blob<Dtype>& blob, int start = 0);
    virtual int read_blockpack_serial(Blob<Dtype>& blob, int start = 0);

    void set_roi(const Blob<Dtype>& blob_in, const vector<Dtype> scales); //use

    ////// Output layer use //////
    inline int pad_h() {
        return _pad_h;
    }
    inline int pad_w() {
        return _pad_w;
    }
    int getroiidbybufferedimgcoords(const int coords_y, const int coords_x);
    void getinputimgcoords(const int roi_id, const Dtype buff_img_y,
            const Dtype buff_img_x, Dtype& input_y, Dtype& input_x);

protected:
    virtual void setinputbuff_cpu(const Blob<Dtype>& blob_in);
    virtual void setinputbuff_gpu(const Blob<Dtype>& blob_in);
    virtual void setblockpackinfo(const Blob<Dtype>& blob_in);

    int _max_stride;
    int _pad_h;
    int _pad_w;

    int _block_width;
    int _block_height;

    float _roi_star_percent;
    float _roi_height_percent;

    NewRectMap _rect_map; //use
    vector<NewRoiRect<Dtype> > _roi; //use
    vector<NewRect> _placed_rect;

    /*  buffers for crop and resize */
    Blob<Dtype> _buff_blob_1; //use
    Blob<Dtype> _buff_blob_2; //use

    Blob<Dtype> _buff_map;
};

///////////////////////////////////////////////////////////////////

}

#endif /* INCLUDE_CAFFE_LAYERS_DL_BLOCKPACKING_HPP_ */
