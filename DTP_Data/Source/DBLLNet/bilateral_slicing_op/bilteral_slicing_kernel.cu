#include "torch/extension.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include <vector>

#include "device_launch_parameters.h"

//Translated from the original HDRNet Tensorflow custom op implementation

__device__ float diff_abs(float x)
{
	float eps = 1e-8;
	return sqrt(x*x + eps);
}

__device__ float d_diff_abs(float x)
{
	float eps = 1e-8;
	return x / sqrt(x*x + eps);
}

__device__ float weight_z(float x)
{
	float abx = diff_abs(x);
	return max(1.0f - abx, 0.0f);
}

__device__ float d_weight_z(float x)
{
	float abx = diff_abs(x);
	if (abx > 1.0f) {
		return 0.0f;
		// return abx;
	}
	else {
		return d_diff_abs(x);
	}
}

struct ExecuteConfig
{
	int nThreadBlock;
	int nThreadPerBlock;
};

ExecuteConfig getExecuteConfig(int size)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int blockSize = deviceProp.maxThreadsPerBlock;
	int nRequiredBlocks = std::min((size + blockSize - 1) / blockSize, //round up
		(deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor + blockSize - 1) / blockSize);
	return { nRequiredBlocks,blockSize };
}

template <typename scalar_t>
__global__ void bilateralSliceApplyCudaForwardKernel
(
	scalar_t* __restrict__ guide,
	scalar_t* __restrict__ input,
	scalar_t* __restrict__ grid,
	scalar_t* __restrict__ output,
	bool  has_offset,
	int h, int w, int bs, int gd, int gh, int gw,
	int input_chans, int output_chans, int outputSize

)
{
	int grid_chans = input_chans * output_chans;
	int coeff_stride = input_chans;
	if (has_offset) {
		grid_chans += output_chans;
		coeff_stride += 1;
	}
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < outputSize; idx += stride)
	{
	/*	int out_c = idx % output_chans;
		int x = (idx / output_chans) % w;
		int y = (idx / (output_chans*w)) % h;
		int b = (idx / (output_chans*w*h));*/
		
		int x = idx % w;
		int y = (idx / w) % h;
		int out_c = (idx / (h*w)) % output_chans;
		int b = (idx / (output_chans*w*h));

		float gx = (x + 0.5f)*gw / (1.0f*w);
		float gy = (y + 0.5f)*gh / (1.0f*h);
		float gz = guide[x + w * (y + h * b)] * gd;

		int fx = static_cast<int>(floor(gx - 0.5f));
		int fy = static_cast<int>(floor(gy - 0.5f));
		int fz = static_cast<int>(floor(gz - 0.5f));


		// Grid strides
		/*int sz = grid_chans;
		int sx = grid_chans * gd;
		int sy = grid_chans * gd*gw;
		int sb = grid_chans * gd*gw*gh;*/
		int sy = gw; 
		int sz = gw * gh; 
		int sc = gw * gh * gd;
		int sb = grid_chans * gd * gw * gh;

		float value = 0.0f;
		for (int in_c = 0; in_c < coeff_stride; ++in_c) {
			float coeff_sample = 0.0f;
			for (int xx = fx; xx < fx + 2; ++xx) {
				int x_ = max(min(xx, gw - 1), 0);
				float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);
				for (int yy = fy; yy < fy + 2; ++yy)
				{
					int y_ = max(min(yy, gh - 1), 0);
					float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);
					for (int zz = fz; zz < fz + 2; ++zz)
					{
						int z_ = max(min(zz, gd - 1), 0);
						float wz = weight_z(zz + 0.5 - gz);
						//int grid_idx = (coeff_stride*out_c + in_c) + sz * z_ + sx * x_ + sy * y_ + sb * b;
						int grid_idx = (coeff_stride*out_c + in_c) *sc + sz * z_ +  x_ + sy * y_ + sb * b;
						coeff_sample += grid[grid_idx] * wx*wy*wz;
					}
				}
			} // Grid trilinear interpolation
			if (in_c < input_chans) {
				//int input_idx = in_c + input_chans * (x + w * (y + h * b));
				int input_idx = x + w * (y + h * (in_c + input_chans * b));
				value += coeff_sample * input[input_idx];
			}
			else { // Offset term
				value += coeff_sample;
			}
		}
		output[idx] = value;
	}
}

template <typename scalar_t>
__global__ void bilateralSliceApplyCudaInputGradKernel
(
	scalar_t* __restrict__ guide,
	scalar_t* __restrict__ input,
	scalar_t* __restrict__ grid,
	scalar_t* __restrict__ backprop,
	scalar_t* __restrict__ input_grad,
	bool  has_offset,
	int h, int w, int bs, int gd, int gh, int gw,
	int input_chans, int output_chans, int inputSize

)
{
	int grid_chans = input_chans * output_chans;
	int coeff_stride = input_chans;
	if (has_offset) {
		grid_chans += output_chans;
		coeff_stride += 1;
	}

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < inputSize; idx += stride)
	{
	/*	int in_c = idx % input_chans;
		int x = (idx / input_chans) % w;
		int y = (idx / (input_chans*w)) % h;
		int b = (idx / (input_chans*w*h));*/
		int x = idx % w;
		int y = (idx / w) % h;
		int in_c = (idx / (h*w)) % output_chans;
		int b = (idx / (output_chans*w*h));

		float gx = (x + 0.5f)*gw / (1.0f*w);
		float gy = (y + 0.5f)*gh / (1.0f*h);
		float gz = guide[x + w * (y + h * b)] * gd;

		int fx = static_cast<int>(floor(gx - 0.5f));
		int fy = static_cast<int>(floor(gy - 0.5f));
		int fz = static_cast<int>(floor(gz - 0.5f));

		// Grid stride 
		//int sz = grid_chans;
		//int sx = grid_chans * gd;
		//int sy = grid_chans * gd*gw;
		//int sb = grid_chans * gd*gw*gh;
		int sy = gw;
		int sz = gw * gh;
		int sc = gw * gh * gd;
		int sb = grid_chans * gd * gw * gh;

		float value = 0.0f;
		for (int out_c = 0; out_c < output_chans; ++out_c) {
			float chan_val = 0.0f;
			for (int xx = fx; xx < fx + 2; ++xx) {
				int x_ = max(min(xx, gw - 1), 0);
				float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);
				for (int yy = fy; yy < fy + 2; ++yy)
				{
					int y_ = max(min(yy, gh - 1), 0);
					float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);
					for (int zz = fz; zz < fz + 2; ++zz)
					{

						int z_ = max(min(zz, gd - 1), 0);

						float wz = weight_z(zz + 0.5 - gz);

						//int grid_idx = (coeff_stride*out_c + in_c) + sz * z_ + sx * x_ + sy * y_ + sb * b;
						int grid_idx = (coeff_stride*out_c + in_c) * sc + sz * z_ + x_ + sy * y_ + sb * b;
						chan_val += grid[grid_idx] * wx*wy*wz;
					} // z
				} // y
			} // x, grid trilinear interp

			//value += chan_val * backprop[out_c + output_chans * (x + w * (y + h * b))];
			value += chan_val * backprop[x + w * (y + h * (out_c + output_chans * b))];
		} // out_c
		input_grad[idx] = value;
	}
}


template <typename scalar_t>
__global__ void bilateralSliceApplyCudaGuideGradKernel
(
	scalar_t* __restrict__ guide,
	scalar_t* __restrict__ input,
	scalar_t* __restrict__ grid,
	scalar_t* __restrict__ backprop,
	scalar_t* __restrict__ guide_grad,
	bool  has_offset,
	int h, int w, int bs, int gd, int gh, int gw,
	int input_chans, int output_chans, int guide_size

)
{
	int grid_chans = input_chans * output_chans;
	int coeff_stride = input_chans;
	if (has_offset) {
		grid_chans += output_chans;
		coeff_stride += 1;
	}

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < guide_size; idx += stride)
	{
		int x = idx % w;
		int y = (idx / w) % h;
		int b = (idx / (w*h));

		float gx = (x + 0.5f)*gw / (1.0f*w);
		float gy = (y + 0.5f)*gh / (1.0f*h);
		float gz = guide[x + w * (y + h * b)] * gd;

		int fx = static_cast<int>(floor(gx - 0.5f));
		int fy = static_cast<int>(floor(gy - 0.5f));
		int fz = static_cast<int>(floor(gz - 0.5f));

		// Grid stride 
		//int sz = grid_chans;
		//int sx = grid_chans * gd;
		//int sy = grid_chans * gd*gw;
		//int sb = grid_chans * gd*gw*gh;
		int sy = gw;
		int sz = gw * gh;
		int sc = gw * gh * gd;
		int sb = grid_chans * gd * gw * gh;

		float out_sum = 0.0f;
		for (int out_c = 0; out_c < output_chans; ++out_c) {

			float in_sum = 0.0f;
			for (int in_c = 0; in_c < coeff_stride; ++in_c) {

				float grid_sum = 0.0f;
				for (int xx = fx; xx < fx + 2; ++xx) {
					int x_ = max(min(xx, gw - 1), 0);
					float wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);
					for (int yy = fy; yy < fy + 2; ++yy)
					{
						int y_ = max(min(yy, gh - 1), 0);
						float wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);
						for (int zz = fz; zz < fz + 2; ++zz)
						{
							int z_ = max(min(zz, gd - 1), 0);
							float dwz = gd * d_weight_z(zz + 0.5 - gz);

							//int grid_idx = (coeff_stride*out_c + in_c) + sz * z_ + sx * x_ + sy * y_ + sb * b;
							int grid_idx = (coeff_stride*out_c + in_c) * sc + sz * z_ + x_ + sy * y_ + sb * b;
							grid_sum += grid[grid_idx] * wx*wy*dwz;
						} // z
					} // y
				} // x, grid trilinear interp

				if (in_c < input_chans) {
				    //in_sum += grid_sum * input[in_c + input_chans * (x + w * (y + h * b))];
					in_sum += grid_sum * input[x + w * (y + h * (in_c + input_chans * b))];
				}
				else {  // offset term
					in_sum += grid_sum;
				}
			} // in_c
			//out_sum += in_sum * backprop[out_c + output_chans * (x + w * (y + h * b))];
			out_sum += in_sum * backprop[x + w * (y + h * (out_c + output_chans * b))];

		} // out_c

		guide_grad[idx] = out_sum;
	}
}

template <typename scalar_t>
__global__ void bilateralSliceApplyCudaGridGradKernel
(
	scalar_t* __restrict__ guide,
	scalar_t* __restrict__ input,
	scalar_t* __restrict__ grid,
	scalar_t* __restrict__ backprop,
	scalar_t* __restrict__ grid_grad,
	bool  has_offset,
	int h, int w, int bs, int gd, int gh, int gw,
	int input_chans, int output_chans, int grid_size

)
{
	int grid_chans = input_chans * output_chans;
	int coeff_stride = input_chans;

	if (has_offset) {
		grid_chans += output_chans;
		coeff_stride += 1;
	}

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < grid_size; idx += stride)
	{
		//int c = idx % grid_chans;
		//int gz = (idx / grid_chans) % gd;
		//int gx = (idx / (grid_chans*gd)) % gw;
		//int gy = (idx / (grid_chans*gd*gw)) % gh;
		//int b = (idx / (grid_chans*gd*gw*gh));
		
		int gx = idx % gw;
		int gy = (idx /gw) % gh;
		int gz = (idx / (gh * gw)) % gd;
		int c = (idx / (gh * gw * gd)) % grid_chans;
		int b = (idx / (grid_chans*gd*gw*gh));

		float scale_w = w * 1.0 / gw;
		float scale_h = h * 1.0 / gh;

		int left_x = static_cast<int>(floor(scale_w*(gx + 0.5 - 1)));
		int right_x = static_cast<int>(ceil(scale_w*(gx + 0.5 + 1)));
		int left_y = static_cast<int>(floor(scale_h*(gy + 0.5 - 1)));
		int right_y = static_cast<int>(ceil(scale_h*(gy + 0.5 + 1)));

		// Strides in the output
		/*int sx = output_chans;
		int sy = output_chans * w;
		int sb = output_chans * w*h;*/
		int sy = w;
		int sc = w * h;
		int sb = output_chans * w * h;

		// Strides in the input
		//int isx = input_chans;
		//int isy = input_chans * w;
		//int isb = input_chans * w*h;
		int isy = w;
		int isc = w * h;
		int isb = input_chans * w * h;

		int out_c = c / coeff_stride;
		int in_c = c % coeff_stride;

		float value = 0.0f;
		for (int x = left_x; x < right_x; ++x)
		{
			int x_ = x;

			// mirror boundary
			if (x_ < 0) x_ = -x_ - 1;
			if (x_ >= w) x_ = 2 * w - 1 - x_;

			float gx2 = (x + 0.5f) / scale_w;
			float wx = max(1.0f - abs(gx + 0.5 - gx2), 0.0f);

			for (int y = left_y; y < right_y; ++y)
			{
				int y_ = y;

				// mirror boundary
				if (y_ < 0) y_ = -y_ - 1;
				if (y_ >= h) y_ = 2 * h - 1 - y_;

				float gy2 = (y + 0.5f) / scale_h;
				float wy = max(1.0f - abs(gy + 0.5 - gy2), 0.0f);

				int guide_idx = x_ + w * y_ + h * w*b;
				float gz2 = guide[guide_idx] * gd;
				float wz = weight_z(gz + 0.5f - gz2);
				if ((gz == 0 && gz2 < 0.5f) || (gz == gd - 1 && gz2 > gd - 0.5f)) {
					wz = 1.0f;
				}

				//int back_idx = out_c + sx * x_ + sy * y_ + sb * b;
				int back_idx = out_c * sc +  x_ + sy * y_ + sb * b;
				if (in_c < input_chans) {
					//int input_idx = in_c + isx * x_ + isy * y_ + isb * b;
					int input_idx = in_c * isc + x_ + isy * y_ + isb * b;
					value += wz * wx*wy*backprop[back_idx] * input[input_idx];
				}
				else { // offset term
					value += wz * wx*wy*backprop[back_idx];
				}
			}
		}
		grid_grad[idx] = value;
	}
}


torch::Tensor bilateralSliceApplyCudaForward
(
	torch::Tensor grid,
	torch::Tensor guide,
	torch::Tensor input,
	bool has_offset,
	torch::Tensor output,
	int h,
	int w,
	int bs,
	int gd,
	int gh,
	int gw,
	int input_chans,
	int output_chans
)
{
	int64_t outputSize = bs * h*w* output_chans;
	ExecuteConfig exConfig = getExecuteConfig(outputSize);

	AT_DISPATCH_FLOATING_TYPES
	(
		input.type(),
		"bilateralSliceApplyCudaForwardKernel",
		(
			[&] {
		bilateralSliceApplyCudaForwardKernel<scalar_t> << <exConfig.nThreadBlock, exConfig.nThreadPerBlock >> >
			(
				guide.data<scalar_t>(),
				input.data<scalar_t>(),
				grid.data<scalar_t>(),
				output.data<scalar_t>(),
				has_offset,
				h, w, bs, gd, gh, gw,
				input_chans, output_chans, outputSize
				)
			; }
			)
	);
	return output;
}

std::vector<torch::Tensor> bilateralSliceApplyCudaBackward
(
	torch::Tensor grid,
	torch::Tensor guide,
	torch::Tensor input,
	torch::Tensor backprop,
	bool has_offset,
	torch::Tensor grid_grad,
	torch::Tensor guide_grad,
	torch::Tensor input_grad,
	int h,
	int w,
	int bs,
	int gd,
	int gh,
	int gw,
	int input_chans,
	int output_chans
)
{
	//Step 1: Compute input image gradient

	int64_t inputSize = bs * h*w* input_chans;
	ExecuteConfig exConfig_input = getExecuteConfig(inputSize);

	if (inputSize > 0)
	{
		AT_DISPATCH_FLOATING_TYPES
		(
			input.type(),
			"bilateralSliceApplyCudaInputGradKernel",
			(
				[&] {
			bilateralSliceApplyCudaInputGradKernel<scalar_t> <<<exConfig_input.nThreadBlock, exConfig_input.nThreadPerBlock >>>
				(
					guide.data<scalar_t>(),
					input.data<scalar_t>(),
					grid.data<scalar_t>(),
					backprop.data<scalar_t>(),
					input_grad.data<scalar_t>(),
					has_offset,
					h, w, bs, gd, gh, gw,
					input_chans, output_chans, inputSize
					)
				; }
				)
		);
	}


	//Step 2: Compute guide map gradient

	int64_t guideSize = bs * h* w;
	ExecuteConfig exConfig_guide = getExecuteConfig(guideSize);

	if (guideSize > 0)
	{
		AT_DISPATCH_FLOATING_TYPES
		(
			guide.type(),
			"bilateralSliceApplyCudaGuideGradKernel",
			(
				[&] {
			bilateralSliceApplyCudaGuideGradKernel<scalar_t> <<<exConfig_guide.nThreadBlock, exConfig_guide.nThreadPerBlock >>>
				(
					guide.data<scalar_t>(),
					input.data<scalar_t>(),
					grid.data<scalar_t>(),
					backprop.data<scalar_t>(),
					guide_grad.data<scalar_t>(),
					has_offset,
					h, w, bs, gd, gh, gw,
					input_chans, output_chans, guideSize
					)
				; }
				)
		);
	}


	//Step 3: Compute bilateral grid gradient

	int64_t gridSize = bs * gh * gw * gd * grid.size(1);
	ExecuteConfig exConfig_grid = getExecuteConfig(gridSize);

	if (gridSize > 0)
	{
		AT_DISPATCH_FLOATING_TYPES(
			grid.type(),
			"bilateralSliceApplyCudaGridGradKernel",
			(
				[&] {
			bilateralSliceApplyCudaGridGradKernel<scalar_t> <<<exConfig_grid.nThreadBlock, exConfig_grid.nThreadPerBlock >>>
				(
					guide.data<scalar_t>(),
					input.data<scalar_t>(),
					grid.data<scalar_t>(),
					backprop.data<scalar_t>(),
					grid_grad.data<scalar_t>(),
					has_offset,
					h, w, bs, gd, gh, gw,
					input_chans, output_chans, gridSize
					)
				; }
				)
		);
	}

	return { input_grad, guide_grad, grid_grad };
}
