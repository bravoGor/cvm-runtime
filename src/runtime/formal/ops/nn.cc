// #include "ops.h"
#include <cvm/runtime/forward.h>
#include <cvm/runtime/registry.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

namespace cvm {
namespace runtime {
CVM_REGISTER_GLOBAL("cvm.runtime.formal.relu")
.set_body([](CVMArgs args, CVMRetValue* rv) {
  auto X_data = CVMArg2Data<int32_t>(args[0]);
  auto Y_data = CVMArg2Data<int32_t>(args[1]);
  const TShape& shp = CVMArgShape(args[0]);

  for (size_t i = 0; i < shp.Size(); ++i) {
    Y_data[i] = std::max(X_data[i], 0);
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {
  // inputs X, W, B
  // outpus Y
  auto X = args[0];
  auto W = args[1];
  auto param = CVMArg2Attr<top::DenseParam>(args[args.size() - 1]);
  auto Y = (param.use_bias) ? args[3] : args[2];

  // X.shape = (M, K)
  // W.shape = (N, K)
  // B.shape = (N,)
  // Y.shape = (M, N)
  auto X_shape = CVMArgShape(X);
  auto W_shape = CVMArgShape(W);
  auto Y_shape = CVMArgShape(Y);

  auto X_data = CVMArg2Data<int32_t>(X);
  auto W_data = CVMArg2Data<int32_t>(W);
  auto Y_data = CVMArg2Data<int32_t>(Y);
  // Y = X * W^T
  Indices xIdx(X_shape), wIdx(W_shape), yIdx(Y_shape);
  for (; !yIdx.End(); yIdx++) {
    // Y(m, n) = X(m, k) * WT(k, n) = X(m, k) * W(n, k)
    int32_t sum = 0;  //, W_offset = n * W_shape[1];
    for (int64_t k = 0; k < X_shape[1]; ++k) {
      xIdx.CopyIndicesFrom({yIdx[0], k});
      wIdx.CopyIndicesFrom({yIdx[1], k});
      sum += X_data[xIdx.Index()] * W_data[wIdx.Index()];
    }
    Y_data[yIdx.Index()] = sum;
  }

  // if B is not None, Y = X * WT + B
  // Y[m, n] += B[n]
  if (param.use_bias) {
    for (Indices yIdx(Y_shape); !yIdx.End(); yIdx++) {
      auto B_data = CVMArg2Data<int32_t>(args[2]);
      Y_data[yIdx.Index()] += B_data[yIdx[1]];
    }
  }
});

static void groupwise_conv2d(
    int32_t* x_data, int32_t n_batch, int32_t in_channels, int32_t x_h,
    int32_t x_w, int32_t* w_data, int32_t filter_c, int32_t filter_h,
    int32_t filter_w, int32_t* y_data, int32_t out_channels, int32_t o_h,
    int32_t o_w, int32_t* b_data, int32_t padding[2], int32_t stride_h,
    int32_t stride_w, int32_t dilation_h, int32_t dilation_w, int32_t groups) {
  int32_t ochannels_per_group = out_channels / groups;
  int32_t ichannels_per_group = in_channels / groups;
#ifdef USE_INDICES
  for (Indices yIdx(TShape({n_batch, out_channels, o_h, o_w})),
       xIdx(TShape{n_batch, in_channels, x_h, x_w}),
       ftrIdx(TShape{out_channels, filter_c, filter_h, filter_w});
       !yIdx.End(); yIdx++) {
    // no need to reset xIdx and ftrIdx(stands for filter index)
    // since they are assigned to a completely new value each time in the loop.
    int n = yIdx[0], oc = yIdx[1], oh = yIdx[2], ow = yIdx[3];
    int32_t sum = 0;
    int32_t ic = oc / ochannels_per_group * ichannels_per_group;
    for (int32_t tic = 0; tic < ichannels_per_group; ++tic) {
      for (int32_t fh = 0; fh < filter_h; ++fh) {
        for (int32_t fw = 0; fw < filter_w; ++fw) {
          int32_t th = oh * stride_h + fh * dilation_h - padding[0];
          int32_t tw = ow * stride_w + fw * dilation_w - padding[1];
          if (th < 0 || tw < 0 || th >= x_h || tw >= x_w) continue;
          xIdx.CopyIndicesFrom({n, ic + tic, th, tw});
          ftrIdx.CopyIndicesFrom({oc, tic, fh, fw});
          sum += x_data[xIdx.Index()] * w_data[ftrIdx.Index()];
        }
      }
    }
    y_data[yIdx.Index()] = sum + (b_data == nullptr ? 0 : b_data[oc]);
  }
#else
  int O_CHW = out_channels * o_h * o_w;
  int O_HW = o_h * o_w;
  int I_CHW = in_channels * x_h * x_w;
  int I_HW = x_h * x_w;
  int F_CHW = filter_c * filter_h * filter_w;
  int F_HW = filter_h * filter_w;
  for (int32_t n = 0; n < n_batch; ++n) {
#pragma omp parallel for collapse(3)
    for (int32_t oc = 0; oc < out_channels; ++oc) {
      for (int32_t oh = 0; oh < o_h; ++oh) {
        for (int32_t ow = 0; ow < o_w; ++ow) {
          int32_t oi =
              n * O_CHW + oc * O_HW + oh * o_w + ow;
          int32_t sum = 0;
          int32_t ic = oc / ochannels_per_group * ichannels_per_group;
          for (int32_t tic = 0; tic < ichannels_per_group; ++tic) {
            for (int32_t fh = 0; fh < filter_h; ++fh) {
              for (int32_t fw = 0; fw < filter_w; ++fw) {
                int32_t th = oh * stride_h + fh * dilation_h - padding[0];
                int32_t tw = ow * stride_w + fw * dilation_w - padding[1];
                if (th < 0 || tw < 0 || th >= x_h || tw >= x_w) continue;
                sum += x_data[n * I_CHW + (ic + tic) * I_HW + th * x_w + tw] *
                       w_data[oc * F_CHW + tic * F_HW + fh * filter_w + fw];
              }
            }
          }
          y_data[oi] = sum + (b_data == nullptr ? 0 : b_data[oc]);
        }
      }
    }
  }
#endif  // USE_INDICES

}


CVM_REGISTER_GLOBAL("cvm.runtime.formal.conv2d")
.set_body([](CVMArgs args, CVMRetValue* rv) {
  auto& param = CVMArg2Attr<top::Conv2DParam>(args[args.num_args - 1]);
  int groups = param.groups;
  int dilation[2] = {(int)param.dilation[0], (int)param.dilation[1]};
  int padding[2] = {(int)param.padding[0], (int)param.padding[1]};
  int strides[2] = {(int)param.strides[0], (int)param.strides[1]};

  int stride_h = strides[0];
  int stride_w = strides[1];

  int32_t* x_data = CVMArg2Data<int32_t>(args[0]);
  int32_t* w_data = CVMArg2Data<int32_t>(args[1]);
  int32_t* y_data =
      CVMArg2Data<int32_t>(param.use_bias ? args[3] : args[2]);
  int32_t* b_data =
      param.use_bias ? CVMArg2Data<int32_t>(args[2]) : nullptr;

  TShape const& xShape = CVMArgShape(args[0]);
  TShape const& wShape = CVMArgShape(args[1]);

  int out_channels = static_cast<int>(wShape[0]);
  int filter_c = static_cast<int>(wShape[1]);
  int filter_h = static_cast<int>(wShape[2]);
  int filter_w = static_cast<int>(wShape[3]);
  int t_filter_h = (filter_h - 1) * dilation[0] + 1;
  int t_filter_w = (filter_w - 1) * dilation[1] + 1;

  int n_batch = static_cast<int>(xShape[0]);
  int in_channels = static_cast<int>(xShape[1]);
  int x_h = static_cast<int>(xShape[2]);
  int x_w = static_cast<int>(xShape[3]);
  int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
  int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;

  groupwise_conv2d(x_data, n_batch, in_channels, x_h, x_w, w_data, filter_c,
                    filter_h, filter_w, y_data, out_channels, o_h, o_w,
                    b_data, padding, stride_h, stride_w, dilation[0],
                    dilation[1], groups);
});

/*
 * strides (2, 2)
 * pool_size [3, 3]
 * ceil_mode False
 * padding (1, 1)
 */
CVM_REGISTER_GLOBAL("cvm.runtime.formal.max_pool2d")
.set_body([](CVMArgs args, CVMRetValue* ret) {
  int32_t* x_data = CVMArg2Data<int32_t>(args[0]);
  int32_t* y_data = CVMArg2Data<int32_t>(args[1]);
  auto& param = CVMArg2Attr<top::MaxPool2DParam>(args[2]);
  TShape const& xShape = CVMArgShape(args[0]);
  TShape const& yShape = CVMArgShape(args[1]);
  int padding[2] = {(int)param.padding[0], (int)param.padding[0]};
  if (param.padding.ndim() == 2) {
    padding[1] = (int)param.padding[1];
  }

  int stride_h = param.strides[0];
  int stride_w = param.strides[1];

  int filter_h = param.pool_size[0];
  int filter_w = param.pool_size[1];

  int n_batch = static_cast<int>(xShape[0]);
  int in_channels = static_cast<int>(xShape[1]);
  int out_channels = in_channels;
  int x_h = static_cast<int>(xShape[2]);
  int x_w = static_cast<int>(xShape[3]);
  int o_h = static_cast<int>(yShape[2]);
  int o_w = static_cast<int>(yShape[3]);
  for (Indices yIdx(yShape), xIdx(xShape); !yIdx.End(); yIdx++) {
    // no need to reset xIdx since content of xIdx assigned completely.
    int n = yIdx[0], k = yIdx[1], p = yIdx[2], q = yIdx[3];
    // 1 << 31 will be the min value for int32: 0x80000000
    const int32_t minV = int32_t(1) << 31;
    int32_t y_max = minV;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        int32_t tp = p * stride_h + r - padding[0];
        int32_t tq = q * stride_w + s - padding[1];
        if (0 <= tp && tp < x_h && 0 <= tq && tq < x_w) {
          // if the region is out of the feature map, y_max remains INT_MIN
          xIdx.CopyIndicesFrom({yIdx[0], yIdx[1], tp, tq});
          y_max = std::max(x_data[xIdx.Index()], y_max);
        }
      }
    }
    y_data[yIdx.Index()] = y_max;
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.upsampling")
.set_body([](CVMArgs args, CVMRetValue* ret) {
  int32_t* x_data = CVMArg2Data<int32_t>(args[0]);
  int32_t* y_data = CVMArg2Data<int32_t>(args[1]);
  auto& param = CVMArg2Attr<top::UpSamplingParam>(args[2]);
  TShape const& xShape = CVMArgShape(args[0]);
  TShape const& yShape = CVMArgShape(args[1]);

  uint32_t scale = {(uint32_t)param.scale};
  uint32_t h = xShape[2], w = xShape[3];
  uint32_t oh = yShape[2], ow = yShape[3];
  uint32_t n_batch = xShape[0], n_channels = xShape[1];

  for (uint32_t batch = 0; batch < n_batch; ++batch) {
    for (uint32_t c = 0; c < n_channels; ++c) {
      auto bc_y_data = y_data + batch * n_channels * oh * ow + c * oh * ow;
      auto bc_x_data = x_data + batch * n_channels * h * w + c * h * w;
      for (uint32_t y = 0; y < oh; ++y) {
        for (uint32_t x = 0; x < ow; ++x) {
          bc_y_data[y * ow + x] = bc_x_data[y / scale * w + x / scale];
        }
      }
    }
  }
});

}
}