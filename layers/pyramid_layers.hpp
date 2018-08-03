#ifndef CAFFE_PYRAMID_LAYERS_HPP_
#define CAFFE_PYRAMID_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "opencv2/opencv.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

#include "caffe/layers/fcn_data_layer.hpp"
#include "caffe/layers/roi_layers.hpp"

#include "caffe/util/RectMap.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_img.hpp"

#include "caffe/blob.hpp"
//#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/internal_thread_old.hpp"
#include "caffe/layer.hpp"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /*Packing and unpacking image blocks.*/
    template<typename Dtype>
    class BlockPacking {
    public:
        BlockPacking();

        virtual ~BlockPacking();

        virtual void SetUpParameter(const BlockPackingParameter &block_packing_param);

        virtual void ImgPacking_cpu(const Blob<Dtype> &blob_in, Blob<Dtype> &blob_out);

        virtual void ImgPacking_gpu(const Blob<Dtype> &blob_in, Blob<Dtype> &blob_out);

        void FeatureMapUnPacking_cpu(const Blob<Dtype> &blob_out, Blob<Dtype> blob_in,
                                     const int num_in_img, int heat_map_a_);

        void FeatureMapUnPacking_gpu(const Blob<Dtype> &blob_out, Blob<Dtype> blob_in,
                                     const int num_in_img, int heat_map_a_);

        virtual int SerializeToBlob(Blob<Dtype> &blob, int start = 0);

        virtual int ReadFromSerialized(Blob<Dtype> &blob, int start = 0);

        void GetFeatureMapStartCoordsByBlockId(const int block_id, const int heat_map_a,
                                               const int heat_map_b, int &map_start_y,
                                               int &map_start_x);
	
        void GetFeatureMapStartCoordsByBlockId(const int block_id, 
            const Dtype heat_map_a, Dtype& map_start_y, 
            Dtype& map_start_x);

        inline void setShowTime(bool show) {
            show_time_ = show;
        }
        inline int num_block_w() {
            return num_block_w_;
        }
        inline int num_block_h() {
            return num_block_h_;
        }
        inline int block_width() {
            return block_width_;
        }
        inline int block_height() {
            return block_height_;
        }
        inline int pad_h() {
            return pad_h_;
        }
        inline int pad_w() {
            return pad_w_;
        }
        inline int max_stride() {
            return max_stride_;
        }
        inline int max_block_size() {
            return max_block_size_;
        }
        inline Blob<Dtype> &buff_map() {
            return buff_map_;
        }

    protected:
        /* Copy blob_in intp buff_map_ for further processing. */
        virtual void SetInputBuff_cpu(const Blob<Dtype> &blob_in);

        virtual void SetInputBuff_gpu(const Blob<Dtype> &blob_in) {
        };

        virtual void SetBlockPackingInfo(const Blob<Dtype> &blob_in);

        void GetBlockingInfo1D(const int in_x, int &out_x, int &out_num);

        int max_stride_;
        int pad_h_;
        int pad_w_;
        int max_block_size_;

        int num_block_w_;
        int num_block_h_;
        int block_width_;
        int block_height_;

        float roi_star_percent;
        float roi_height_percent;

        bool show_time_;
        /**
         * @warning  buff_map_ might be very large!
         * So don't synchronize this variable to gpu
         * memory
         */
        Blob<Dtype> buff_map_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Dtype>
    class RoiRect {
    public:
        RoiRect(Dtype scale = 0, Dtype start_y = 0, Dtype start_x = 0, Dtype height = 0,
                Dtype width = 0);

        ~RoiRect() {
        };
        inline Dtype GetArea() const {
            return height_ * width_;
        }
        inline Dtype GetScaledArea() const {
            return scale_ * height_ * scale_ * width_;
        }
        inline Dtype GetScaledHeight() const {
            return scale_ * height_;
        }
        inline Dtype GetScaledWidth() const {
            return scale_ * width_;
        }
        inline Dtype GetScaledX(Dtype dx) const {
            return (start_x_ + dx) * scale_;
        }
        inline Dtype GetScaledY(Dtype dy) const {
            return (start_y_ + dy) * scale_;
        }
        inline Dtype GetOriX(Dtype scaled_dx) const {
            return start_x_ + scaled_dx / scale_;
        }
        inline Dtype GetOriY(Dtype scaled_dy) const {
            return start_y_ + scaled_dy / scale_;
        }
        static bool greaterScale(const RoiRect &a, const RoiRect &b) {
            return a.scale_ > b.scale_;
        }
        static bool greaterScaledArea(const RoiRect &a, const RoiRect &b) {
            return a.GetScaledArea() > b.GetScaledArea();
        }
        static bool greaterMaxScaledEdge(const RoiRect &a, const RoiRect &b) {
            return MAX(a.GetScaledHeight(), a.GetScaledWidth()) >
                   MAX(b.GetScaledHeight(), b.GetScaledWidth());
        }
        friend ostream &operator<<(ostream &stream, RoiRect &rect) {
            stream << "(" << rect.scale_ << "," << rect.start_y_ << "," << rect.start_y_ << ","
                   << rect.height_ << "," << rect.width_ << ")";
            return stream;
        }

        int SerializeToBlob(Blob<Dtype> &blob, int start = 0);

        int ReadFromSerialized(Blob<Dtype> &blob, int start = 0);

        Dtype scale_;
        Dtype start_y_;
        Dtype start_x_;
        Dtype height_;
        Dtype width_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     * Packing and unpacking image blocks. This class enables PatchWorks over scales on the input blob.
     * Unlike the BlobkPacking class, the input blob should have num() == 1.
     */
    template<typename Dtype>
    class RectBlockPacking : public BlockPacking<Dtype> {
    public:
        RectBlockPacking() {
        };
        virtual ~RectBlockPacking() {
        };

        ///call this function before packing
        void setRoi(const Blob<Dtype> &blob_in, const vector<Dtype> scales);

        void setRoi(const Blob<Dtype> &blob_in, const pair<string, vector<Dtype> > &cur_sample);

        virtual int SerializeToBlob(Blob<Dtype> &blob, int start = 0);

        virtual int ReadFromSerialized(Blob<Dtype> &blob, int start = 0);

        int GetRoiIdByBufferedImgCoords(const int coords_y, const int coords_x);

        void GetInputImgCoords(const int roi_id, const Dtype buff_img_y, const Dtype buff_img_x,
                               Dtype &input_y, Dtype &input_x);

    protected:
        virtual void SetBlockPackingInfo(const Blob<Dtype> &blob_in);

        virtual void SetInputBuff_cpu(const Blob<Dtype> &blob_in);

        virtual void SetInputBuff_gpu(const Blob<Dtype> &blob_in);

        /// serialize Rect to blob;
        int SerializeToBlob(Blob<Dtype> &blob, Rect &rect, int start = 0);

        int ReadFromSerialized(Blob<Dtype> &blob, Rect &rect, int start = 0);

        RectMap rectMap_;
        vector<RoiRect<Dtype> > roi_;
        vector<Rect> placedRect_;
        /* buffers for crop and resize */
        Blob<Dtype> buff_blob_1_;
        Blob<Dtype> buff_blob_2_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /*The same as parameters in PyramidImageDataLayer. This struct is used for parameter passing.*/
    template<typename Dtype>
    struct PyramidImageDataParam {
        int ReadFromSerialized(Blob<Dtype> &blob, int start = 0);

        inline int GetBlockIdBy(const int num_id) {
            return forward_iter_id_ * max_block_num_ + num_id;
        }
        int img_w_, img_h_;
        int heat_map_a_;
        int heat_map_b_;
        int max_block_num_;
        RectBlockPacking<Dtype> rect_block_packer_;
        int forward_times_for_cur_sample_;
        int forward_iter_id_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Dtype>
    class PyramidImageDataLayer : public Layer<Dtype>, public InternalThread {
    public:
        explicit PyramidImageDataLayer(const LayerParameter &param) : Layer<Dtype>(param) {
        }

        virtual ~PyramidImageDataLayer();

        virtual void
        LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual void
        Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        };

        virtual inline const char *type() const {
            return "PyramidImageData";
        }
        virtual inline int ExactNumBottomBlobs() const {
            return 0;
        }
        virtual inline int ExactNumTopBlobs() const {
            return 2;
        }

        /* top[0] contains the blocked image,
         * and top[1] pass the PyramidImageDataParam to next layer. */
        virtual void
        Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

#ifdef CPU_ONLY
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top) {}
#else

        virtual void
        Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

#endif
        virtual void
        Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }
        virtual void
        Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }

        pair<string, vector<Dtype> > GetCurSample();

        int GetTotalSampleSize();

        int GetCurSampleID();

        int GetForwardTimesForCurSample();

        int GetCurForwardIDForCurSample();

    protected:
        virtual void CreatePrefetchThread();

        virtual void JoinPrefetchThread();

        //virtual void InternalThreadEntryOld();
        virtual void InternalThreadEntry();

        virtual int SerializeToBlob(Blob<Dtype> &blob, int start = 0);

        void ShowImg(const vector<Blob<Dtype> *> &top);

        ImageDataSourceProvider<Dtype> samples_provider_;

        bool shuffle_;

	      Dtype img_w_[2];
	      Dtype img_h_[2];
        int heat_map_a_;
        int heat_map_b_;
        Dtype mean_bgr_[3];
        Dtype scale_start_;
        Dtype scale_end_;
        Dtype scale_step_;
        bool scale_from_annotation_;
        int max_block_num_;

        pair<string, vector<Dtype> > cur_sample_1_;
        pair<string, vector<Dtype> > cur_sample_2_;
        vector<pair<string, vector<Dtype> > *> cur_sample_list_;

        RectBlockPacking<Dtype> rect_block_packer_1_;
        RectBlockPacking<Dtype> rect_block_packer_2_;
        vector<RectBlockPacking<Dtype> *> rect_block_packer_list_;

        Blob<Dtype> img_blob_1_;
        Blob<Dtype> img_blob_2_;
        vector<Blob<Dtype> *> img_blob_list_;

        vector<Blob<Dtype> *> buffered_block_;
        Blob<Dtype> buffered_block_1_;
        Blob<Dtype> buffered_block_2_;

        int used_buffered_block_id_;

        int forward_times_for_cur_sample_;
        int forward_iter_id_;
        int cur_sample_id_;

        string show_output_path_;
        bool pic_print_;
        bool show_time_;
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Dtype>
    class PyramidImageOnlineDataLayer : public PyramidImageDataLayer<Dtype> {
    public:
        explicit PyramidImageOnlineDataLayer(const LayerParameter &param)
                : PyramidImageDataLayer<Dtype>(param) {
        }
        virtual ~PyramidImageOnlineDataLayer() {
        };

        virtual void
        LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const {
            return "PyramidImageOnlineData";
        }

        /*top[0] contains the blocked image,
         * and top[1] pass the PyramidImageDataParam to next layer.*/
        virtual void
        Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

#ifdef CPU_ONLY
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top) {}
#else

        virtual void
        Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

#endif
        virtual void
        Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }
        virtual void
        Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }

        int GetTotalSampleSize();

        void LoadOneImgToInternalBlob(const Blob<Dtype> &src);

        bool use_video;
        cv::Mat CurDetMat;
        cv::Mat OutDetMat;

        // sued for outter image data

        cv::Mat CurRosMat;
        cv::Mat OutRosMat;

        void FetchOutterImageFrame(cv::Mat &rosImage);

        cv::String video_file;
        cv::VideoCapture VideoCap;

        cv::String list_file;
        std::ifstream bb_file;

        float resize_scale;
    protected:
        virtual void CreatePrefetchThread();//{}
        virtual void JoinPrefetchThread();//{}
        //virtual void InternalThreadEntryOld();
        virtual void InternalThreadEntry();

        void ShowImg(const vector<Blob<Dtype> *> &top);
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Dtype>
    class DetectionOutputLayer : public ROIOutputLayer<Dtype> {
    public:
        explicit DetectionOutputLayer(const LayerParameter &param) : ROIOutputLayer<Dtype>(param) {
        }
        virtual ~DetectionOutputLayer() {
        };

        virtual void
        LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);//check

        virtual inline const char *type() const {
            return "DetectionOutput";
        }
        virtual inline int ExactNumBottomBlobs() const {
            return 2;
        }
        virtual inline int ExactNumTopBlobs() const {
            return 0;
        }

        virtual void
        Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual void
        Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual void
        Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }
        virtual void
        Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                     const vector<Blob<Dtype> *> &bottom) {
        }

    protected:
        PyramidImageDataParam<Dtype> pyramid_image_data_param_;
        bool refine_out_of_map_bbox_;
    };

    // added by mingli
    // Deprecated, replaced by ProposalSSDLayer
    template <typename Dtype>
    class DetectionOutputSSDLayer : public ROIOutputSSDLayer <Dtype>
    {
        public:
            explicit DetectionOutputSSDLayer(const LayerParameter& param) : ROIOutputSSDLayer<Dtype>(param) {
            }
            virtual ~DetectionOutputSSDLayer() {
            };
            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top) {
            };

            virtual inline const char* type() const { 
                return "DetectionOutputSSD"; 
            }

            virtual inline int MinBottomBlobs() const { 
                return 2; 
            }
            virtual inline int MaxBottomBlobs() const { 
                return -1; 
            }
            virtual inline int ExactNumTopBlobs() const { 
                return -1; 
            }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);//check
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);//check

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, 
                    const vector<Blob<Dtype>*>& bottom) {
            }
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, 
                    const vector<Blob<Dtype>*>& bottom) {
            }

        protected:
            PyramidImageDataParam<Dtype> pyramid_image_data_param_;
    };

    template <typename Dtype>
    class ProposalSSDLayer : public ROIOutputSSDLayer <Dtype>
    {
        public:
            explicit ProposalSSDLayer(const LayerParameter& param) : 
                    ROIOutputSSDLayer<Dtype>(param) {}
            virtual ~ProposalSSDLayer(){};

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual inline const char* type() const { return "ProposalSSD"; }
            virtual inline int MinBottomBlobs() const { return 1; }
            virtual inline int MaxBottomBlobs() const { return -1; }
            virtual inline int MinTopBlobs() const { return 1; }
            virtual inline int MaxTopBlobs() const { return -1; }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
        protected:
            int num_ssd_;
            bool has_img_info_;
            int rois_dim_;
            PyramidImageDataParam<Dtype> pyramid_image_data_param_;
    };

    template <typename Dtype>
    class RPNProposalSSDLayer : public ROIOutputSSDLayer <Dtype>
    {
        public:
            explicit RPNProposalSSDLayer(const LayerParameter& param) : 
                ROIOutputSSDLayer<Dtype>(param) {}
            virtual ~RPNProposalSSDLayer(){};

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual inline const char* type() const { return "RPNProposalSSD"; }
            virtual inline int MinBottomBlobs() const { return 2; }
            virtual inline int MaxBottomBlobs() const { return -1; }
            virtual inline int MinTopBlobs() const { return 0; }
            virtual inline int MaxTopBlobs() const { return -1; }
            virtual inline int  ExactNumTopBlobs() const { return -1; }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

        protected:
            int num_rpns_;
            int num_anchors_;
            bool has_img_info_;
            int rois_dim_;
            PyramidImageDataParam<Dtype> pyramid_image_data_param_;
    };

    template <typename Dtype>
    class RCNNProposalLayer : public ROIOutputSSDLayer <Dtype>
    {
        public:
            explicit RCNNProposalLayer(const LayerParameter& param) : 
                    ROIOutputSSDLayer<Dtype>(param) {}
            virtual ~RCNNProposalLayer(){};

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual inline const char* type() const { return "RCNNProposal"; }
            virtual inline int MinBottomBlobs() const { return 3; }
            virtual inline int MaxBottomBlobs() const { return 4; }
            virtual inline int MinTopBlobs() const { return 0; }
            virtual inline int MaxTopBlobs() const { return -1; }
            virtual inline int  ExactNumTopBlobs() const { return -1; }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

        protected:
            bool has_img_info_;
            int rois_dim_;
            PyramidImageDataParam<Dtype> pyramid_image_data_param_;
    };

    template <typename Dtype>
    class RCNNDetOutputWithAttrLayer : public ROIOutputSSDLayer <Dtype>
    {
        public:
            explicit RCNNDetOutputWithAttrLayer(const LayerParameter& param) : 
                    ROIOutputSSDLayer<Dtype>(param) {}
            virtual ~RCNNDetOutputWithAttrLayer(){};

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual inline const char* type() const { return "RCNNDetOutputWithAttr"; }
            virtual inline int MinBottomBlobs() const { return 1; }
            virtual inline int MaxBottomBlobs() const { return -1; }
            virtual inline int MinTopBlobs() const { return 0; }
            virtual inline int MaxTopBlobs() const { return 0; }
            virtual inline int  ExactNumTopBlobs() const { return 0; }

            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                    const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, 
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

        protected:
            bool has_img_info_;
            int num_rois_;
            int rois_dim_;
            int num_kpts_;
            int kpts_cls_dim_;
            int kpts_reg_dim_;
            int num_atrs_;
            int num_ftrs_;
            int num_spmp_;
            int num_cam3d_;
            PyramidImageDataParam<Dtype> pyramid_image_data_param_;
    };
    // end mingli

}  // namespace caffe

#endif  // CAFFE_PYRAMID_DATA_LAYERS_HPP_

