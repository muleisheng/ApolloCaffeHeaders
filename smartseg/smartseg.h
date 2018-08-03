#pragma once

#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// #include <arpa/inet.h>
// #include <errno.h>
// #include <fcntl.h>
// #include <malloc.h>
// #include <netdb.h>
// #include <net/if.h>
// #include <netinet/in.h>
// #include <pthread.h>
// #include <semaphore.h>
// #include <signal.h>
// #include <stdio_ext.h>
// #include <sys/ioctl.h>
// #include <sys/resource.h>
// #include <sys/socket.h>
// #include <sys/socket.h>
// #include <sys/stat.h>
// #include <sys/syscall.h>
// #include <sys/types.h>
// #include <sys/wait.h>
// #include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/noncopyable.hpp>

#include <omp.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Eigen>

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>
#include <caffe/proto/caffe.pb.h>

using caffe::Blob;

#include "basic.h"
#include "random.h"
#include "channel.h"
#include "disjoint_set.h"
#include "timer.h"

#include "common.h"
#include "point_cloud.h"
#include "config.h"
#include "config_layer.h"
#include "frame.h"
#include "frame_layer.h"
#include "frame_data_layer.h"
#include "frame_sequence.h"
#include "frame_sequence_data_layer.h"
#include "blob_visualize_layer.h"

#include "base_view.h"
#include "base_extractor_layer.h"

#include "plan_view.h"
#include "plan_view_layer.h"
#include "plan_view_extractor_layer.h"
#include "plan_view_point_extractor_layer.h"

#include "horizontal_view.h"
#include "horizontal_view_layer.h"
#include "horizontal_view_extractor_layer.h"

#include "side_view.h"
#include "side_view_layer.h"
#include "side_view_extractor_layer.h"

#include "view_pooling_layer.h"
#include "view_unpooling_layer.h"

#include "point_interpolation_layer.h"

#include "segmentor.h"
#include "segmentor2.h"
#include "segmentor_layer.h"

#include "frame_target.h"
#include "frame_target2.h"
#include "frame_target_layer.h"
