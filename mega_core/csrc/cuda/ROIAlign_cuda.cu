#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel loop macro
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data, const int height, const int width,
    T y, T x, const int index) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  y = max(y, T(0));
  x = max(x, T(0));

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high = min(y_low + 1, height - 1);
  int x_high = min(x_low + 1, width - 1);

  T ly = y - y_low, lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];

  return (hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4);
}

template <typename T>
__global__ void RoIAlignForward(
    const int nthreads, const T* bottom_data, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio, const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int roi_bin_grid_h = sampling_ratio > 0
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w = sampling_ratio > 0
                             ? sampling_ratio
                             : ceil(roi_width / pooled_width);
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      T y = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        T x = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;
        output_val += bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
      }
    }
    top_data[index] = output_val / count;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x,
    T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high, const int index) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  y = max(y, T(0));
  x = max(x, T(0));

  y_low = (int)y;
  x_low = (int)x;
  y_high = min(y_low + 1, height - 1);
  x_high = min(x_low + 1, width - 1);

  T ly = y - y_low, lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  w1 = hy * hx;
  w2 = hy * lx;
  w3 = ly * hx;
  w4 = ly * lx;
}

template <typename T>
__global__ void RoIAlignBackwardFeature(
    const int nthreads, const T* top_diff, const int num_rois,
    const T spatial_scale, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio, T* bottom_diff, const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;

    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    int roi_bin_grid_h = sampling_ratio > 0
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w = sampling_ratio > 0
                             ? sampling_ratio
                             : ceil(roi_width / pooled_width);
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      T y = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        T x = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        }
      }
    }
  }
}

// Host interfaces

at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input, const at::Tensor& rois, float spatial_scale,
    int pooled_height, int pooled_width, int sampling_ratio) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options());
  int output_size = num_rois * channels * pooled_height * pooled_width;

  const int threads = 512;
  const int blocks = (output_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ROIAlign_forward_cuda", [&] {
    RoIAlignForward<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output_size,
        input.contiguous().data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        rois.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad, const at::Tensor& rois,
    float spatial_scale, int pooled_height, int pooled_width,
    int batch_size, int channels, int height, int width,
    int sampling_ratio) {
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());
  int grad_size = grad.numel();

  const int threads = 512;
  const int blocks = (grad_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "ROIAlign_backward_cuda", [&] {
    RoIAlignBackwardFeature<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_size,
        grad.contiguous().data_ptr<scalar_t>(),
        rois.size(0),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        grad_input.data_ptr<scalar_t>(),
        rois.contiguous().data_ptr<scalar_t>());
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}
