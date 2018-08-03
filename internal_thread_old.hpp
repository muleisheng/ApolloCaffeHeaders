#ifndef CAFFE_INTERNAL_THREAD_OLD_HPP_
#define CAFFE_INTERNAL_THREAD_OLD_HPP_

#include "caffe/common.hpp"

/*Forward declare boost::thread instead of including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010) on OSX.*/
namespace boost { class thread; }

namespace caffe
{
/*Virtual class encapsulate boost::thread for use in base class The child class will acquire the ability to run a single thread, by reimplementing the virutal function InternalThreadEntry. */
class InternalThreadOld
{
 public:
  InternalThreadOld() : thread_() {}
  virtual ~InternalThreadOld();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThreadOld();

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExitOld();

  bool is_startedOld() const;

 protected:
  /* Implement this method in your subclass with the code you want your thread to run. */
  virtual void InternalThreadEntryOld() {}

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
