import torch
import bilateral_slicing

class Slicing_Apply_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, guide, frinput):
        has_offset = True;
        output = bilateral_slicing.forward(grid, guide, frinput, has_offset)
        ctx.save_for_backward(grid, guide, frinput)
        ctx.offset = has_offset
        return output

    @staticmethod
    def backward(ctx, grad):
        grid, guide, frinput = ctx.saved_tensors
        has_offset = ctx.offset
        outputs = bilateral_slicing.backward(grid, guide, frinput, grad, has_offset)
        grad_grid, grad_guide, grad_frinput = outputs
        return grad_frinput, grad_guide, grad_grid, None
