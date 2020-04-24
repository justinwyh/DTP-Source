#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

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
);

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
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DIMENSION(x,s) TORCH_CHECK(x.dim() == s, #x " should be in " #s "D")

 torch::Tensor bilateral_slice_forward
 (
	torch::Tensor grid,
	torch::Tensor guide,
	torch::Tensor input,
	bool has_offset
 ) 
{
	CHECK_INPUT(grid);
	CHECK_INPUT(guide);
	CHECK_INPUT(input);

	CHECK_DIMENSION(grid, 5);
	CHECK_DIMENSION(guide, 3);
	CHECK_DIMENSION(input, 4);

	int64_t h = guide.size(1);
	int64_t w = guide.size(2);
	int64_t bs = grid.size(0);
	int64_t gh = grid.size(2);
	int64_t gw = grid.size(3);
	int64_t gd = grid.size(4);
	int64_t coeffs_chans = grid.size(1);
	int64_t input_chans = input.size(1);
	
	TORCH_CHECK((input.size(0) == guide.size(0)) && (input.size(2) == h) && (input.size(3) == w), "Input and guide size should match");
	TORCH_CHECK((guide.size(0) == bs), "Batch sizes should match");

	int output_chans = 0;
	if (has_offset) 
	{
		TORCH_CHECK((coeffs_chans % (input_chans + 1) == 0), "Slicing with affine offset, coefficients grid should have n_out*(n_in+1) channels");
		output_chans = coeffs_chans / (input_chans + 1);
	}
	else 
	{
		TORCH_CHECK((coeffs_chans % input_chans == 0), "Slicing without affine offset, coefficients grid should have n_out*n_in channels");
		output_chans = coeffs_chans / input_chans;
	}

	//Allocate output tensor
	torch::Tensor output = torch::empty({ bs,output_chans, h, w }, input.options()); //fit TensorOptions of input to output //==Tensor.type() (deprecated)
	return bilateralSliceApplyCudaForward(grid, guide, input, has_offset, output, h, w, bs, gd, gh, gw, input_chans, output_chans);
}

std::vector<torch::Tensor> bilateral_slice_backward
(
	torch::Tensor grid,
	torch::Tensor guide,
	torch::Tensor input,
	torch::Tensor backprop,
	bool has_offset
) 
{
	CHECK_INPUT(grid);
	CHECK_INPUT(guide);
	CHECK_INPUT(input);
	CHECK_INPUT(backprop);

	CHECK_DIMENSION(grid, 5);
	CHECK_DIMENSION(guide, 3);
	CHECK_DIMENSION(input, 4);
	CHECK_DIMENSION(input, 4);

	int64_t h = guide.size(1);
	int64_t w = guide.size(2);
	int64_t bs = grid.size(0);
	int64_t gh = grid.size(2);
	int64_t gw = grid.size(3);
	int64_t gd = grid.size(4);
	int64_t coeffs_chans = grid.size(1);
	int64_t input_chans = input.size(1);

	int output_chans = 0;
	if (has_offset)
	{
		output_chans = coeffs_chans / (input_chans + 1);
	}
	else
	{
		output_chans = coeffs_chans / input_chans;
	}

	//Allocate output tensor
	torch::Tensor grid_grad = torch::empty(grid.sizes(), grid.options());
	torch::Tensor guide_grad = torch::empty(guide.sizes(), guide.options());
	torch::Tensor input_grad = torch::empty(input.sizes(), input.options());

	return bilateralSliceApplyCudaBackward(grid,guide,input,backprop,has_offset,grid_grad,guide_grad,input_grad,h,w,bs,gd,gh,gw,input_chans,output_chans);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
	m.def("forward", &bilateral_slice_forward, "Bilateral grid slicing forward (CUDA)");
	m.def("backward", &bilateral_slice_backward, "Bilateral grid slicing backward (CUDA)");
}