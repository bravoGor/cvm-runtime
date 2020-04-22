#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>

#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "ops.h"

namespace cvm{
  namespace runtime{

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
    size *= dlTensor->shape[i];
  }
  return size;
}

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.elemwise_add")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *a = args[0];
      DLTensor *b = args[1];
      DLTensor *c = args[2];
      void *a_data = (a->data);
      void *b_data = (b->data);
      void *c_data = (c->data);
      uint64_t n = getSize(a);
      //int error_code = 0;
      opencl_elemwise_add(a_data, b_data, c_data, n);

});

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.conv2d")
  .set_body([](CVMArgs args, CVMRetValue* rv){
      DLTensor *x = args[0];
      DLTensor *w = args[1];
      DLTensor *b = nullptr;
      DLTensor *y = nullptr;
      void *_attr;

      if(args.num_args == 5){
      b = args[2];
      y = args[3];
      _attr = args[4];
      } else {
      y = args[2];
      _attr = args[3];
      }
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::Conv2DParam>(attr->parsed);
      //int groups = param.groups;
      int dilation[2] = {static_cast<int>(param.dilation[0]), static_cast<int>(param.dilation[1])};
      int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[1])};
      int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};
      bool use_bias = param.use_bias;


      void* x_data = x->data;
      void* w_data = w->data;
      void* y_data = y->data;
      void* b_data = b != nullptr ? b->data : nullptr;

      int out_channels = static_cast<int>(w->shape[0]);
      //int filter_c = static_cast<int>(w->shape[1]);
      int filter_h = static_cast<int>(w->shape[2]);
      int filter_w = static_cast<int>(w->shape[3]);
      int t_filter_h = (filter_h - 1) * dilation[0] + 1;
      int t_filter_w = (filter_w - 1) * dilation[1] + 1;

      int n_batch = static_cast<int>(x->shape[0]);
      int in_channels = static_cast<int>(x->shape[1]);
      int x_h = static_cast<int>(x->shape[2]);
      int x_w = static_cast<int>(x->shape[3]);
      int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
      int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;

      void *ext_space = args.ext_space->data;
      int32_t ext_space_size = args.ext_space->shape[0];
      opencl_conv2d(x_data, w_data, b_data, y_data, 
          n_batch, in_channels, x_h, x_w, 
          out_channels, filter_h, filter_w, 
          o_h, o_w, 
          padding[0], padding[1],
          strides[0], strides[1],
          dilation[0], dilation[1],
          use_bias,
          ext_space, ext_space_size);

      //int error_code = NON_ERROR;
      //const char* errorStr = "";
      //if(groups == 1){
      //  int32_t *ext_space = static_cast<int32_t*>(args.ext_space->data);
      //  int32_t ext_space_size = args.ext_space->shape[0];
      //  errorStr = opencl_conv2d(
      //      x_data, n_batch, in_channels, x_h, x_w,
      //      w_data, out_channels, in_channels, filter_h, filter_w,
      //      b_data,
      //      padding[0], padding[1],
      //      strides[0], strides[1],
      //      dilation[0], dilation[1],
      //      groups,
      //      y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, 
      //      ext_space,
      //      ext_space_size,
      //      error_code);
      //}else{
      //  errorStr = opencl_groupwise_conv2d(
      //      x_data, n_batch, in_channels, x_h, x_w,
      //      w_data, out_channels, filter_c, filter_h, filter_w,
      //      b_data,
      //      padding[0], padding[1],
      //      strides[0], strides[1],
      //      dilation[0], dilation[1],
      //      groups,
      //      y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
      //}
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.dense")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
      int ndim = args.num_args;
      DLTensor *x = args[0];
      DLTensor *w = args[1];
      DLTensor *bias = nullptr;
      DLTensor *y = nullptr;
      void* bias_data = nullptr;
      void* _attr;
      if(ndim == 5){
        bias = args[2];
        y = args[3];
        bias_data = bias->data;
        _attr = args[4];
      } else{
        y = args[2];
        _attr = args[3];
      }

      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::DenseParam>(attr->parsed);
      void* x_data = x->data;
      void* y_data = y->data;
      void* w_data = w->data;
      bool use_bias = param.use_bias;
      opencl_dense(x_data, w_data, bias_data, y_data, y->shape[0], y->shape[1], x->shape[1], use_bias);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_dense(
      //    x_data, w_data, y_data,
      //    static_cast<int32_t>(x->shape[0]),
      //    static_cast<int32_t>(x->shape[1]),
      //    static_cast<int32_t>(y->shape[1]),
      //    bias_data,
      //    error_code);

      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.relu")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      opencl_relu(x->data, y->data, getSize(x));
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_relu(
      //    static_cast<int32_t*>(x->data),
      //    static_cast<int32_t*>(y->data),
      //    getSize(x),
      //    error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.flatten")
  .set_body([](CVMArgs args, CVMRetValue* rv){
      DLTensor *x = args[0];
      DLTensor *y = args[1];

      opencl_flatten(x->data, y->data, getSize(x));
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_flatten(
      //    static_cast<int32_t*>(x->data),
      //    static_cast<int32_t*>(y->data),
      //    getSize(x),
      //    error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.broadcast_mul")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *args0 = args[0];
      DLTensor *args1 = args[1];
      DLTensor *args2 = args[2];
      void *a = args0->data;
      void *b = args1->data;
      void *c = args2->data;
      //int64_t *ashape = static_cast<int64_t*>(args0->shape);
      //int32_t adim = static_cast<int32_t>(args0->ndim);
      int64_t *bshape = static_cast<int64_t*>(args1->shape);
      int32_t bdim = static_cast<int32_t>(args1->ndim);
      //int64_t *cshape = static_cast<int64_t*>(args2->shape);
      //int32_t cdim = static_cast<int32_t>(args2->ndim);

      if(bdim == 1 && bshape[0] == 1)
        opencl_broadcast_mul(a, b, c, getSize(args0));
      else printf("no support b dim > 1\n");
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_broadcast_mul(a, b, c, getSize(args2),
      //    ashape, adim,
      //    bshape, bdim,
      //    cshape, cdim, error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.max_pool2d")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
      int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};
      int pool_size[2] = {static_cast<int>(param.pool_size[0]), static_cast<int>(param.pool_size[1])};
      int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[0])};
      if(param.padding.ndim() == 2){
        padding[1] = static_cast<int>(param.padding[1]);
      }
      // bool ceil_mode = param.ceil_mode;

      int stride_h = strides[0];
      int stride_w = strides[1];

      void* x_data = x->data;
      void* y_data = y->data;

      int filter_h = pool_size[0];
      int filter_w = pool_size[1];

      int n_batch = static_cast<int>(x->shape[0]);
      int in_channels = static_cast<int>(x->shape[1]);
      //int out_channels = in_channels;
      int x_h = static_cast<int>(x->shape[2]);
      int x_w = static_cast<int>(x->shape[3]);
      //  int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
      //  int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
      int o_h = static_cast<int>(y->shape[2]);
      int o_w = static_cast<int>(y->shape[3]);

      opencl_max_pool2d(x_data, y_data, 
          n_batch, in_channels, x_h, x_w,
          filter_h, filter_w,
          o_h, o_w,
          padding[0], padding[1],
          stride_h, stride_w);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_max_pool(
      //    x_data, n_batch, in_channels, x_h, x_w,
      //    filter_h, filter_w,
      //    padding[0], padding[1],
      //    stride_h, stride_w,
      //    y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_clip")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *x = args[0];
      DLTensor *y = args[1];
      void *x_data = x->data;
      void *y_data = y->data;
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
      int32_t precision = param.precision;

      opencl_cvm_clip(x_data, y_data, getSize(x), precision);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_cvm_clip(
      //    x_data,
      //    precision,
      //    y_data,
      //    getSize(x),
      //    error_code);
      //deal_error(error_code, errorStr);
  });

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.cvm_right_shift")
  .set_body([](CVMArgs args, CVMRetValue *ret){
      DLTensor *a = args[0];
      DLTensor *c = args[1];
      void *_attr = args[2];
      auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
      auto &param = cvm::get<cvm::top::CVMRightShiftParam>(attr->parsed);
      int32_t precision = param.precision;
      int32_t b = param.shift_bit;
      void* a_data = a->data;
      void* c_data = c->data;

      opencl_cvm_right_shift(a_data, c_data, b, getSize(a), precision);
      //int error_code = NON_ERROR;
      //const char* errorStr = opencl_cvm_right_shift(
      //    a_data,
      //    b,
      //    precision,
      //    c_data,
      //    getSize(a),
      //    error_code);
      //deal_error(error_code, errorStr);
  });
}
}