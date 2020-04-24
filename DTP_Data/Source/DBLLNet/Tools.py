import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from DBLLNet.bilateral_slicing_op.bsliceapply import Slicing_Apply_Function

from torch.autograd import gradcheck

from pytorch_msssim import ssim, msssim


grid = torch.rand(2, 12, 8, 3, 3,dtype=torch.double, requires_grad=True).cuda()
guide =  torch.rand(2, 3, 3, dtype=torch.double, requires_grad=True).cuda()
frinput =  torch.rand(2, 3, 3, 3,dtype=torch.double, requires_grad=True).cuda()


def grad_test(grid, guide, frinput):
  return Slicing_Apply_Function.apply(grid, guide, frinput)

is_grad_correct = gradcheck(grad_test, [grid, guide, frinput], eps=1e-3, atol=1e1, raise_exception=True)
print(is_grad_correct)

def showIMG(imgs):
  for img in imgs:
    plt.figure()
    if img.shape[0] == 3:
      img_np = img.permute(1,2,0).cpu().detach().numpy()
      print("showIMG() Output",img_np.shape)
      plt.imshow(img_np,interpolation="bilinear")
    elif img.shape[0] == 1:
      g_out_np = img.permute(1,2,0).cpu().detach().squeeze(2).numpy()
      print("showIMG() Guide Map",g_out_np.shape)
      plt.imshow(g_out_np,cmap='Accent_r',interpolation="bilinear")

def loadIMG(image_path):
  fr_img_loader = transforms.Compose([transforms.ToTensor()])
  lr_img_loader = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
  image = Image.open(image_path)
  fr_image = fr_img_loader(image)
  lr_image = lr_img_loader(image)
  print(fr_image.shape)
  fr_image = fr_image.unsqueeze(0)
  lr_image = lr_image.unsqueeze(0)
  return lr_image, fr_image


def compute_PSNR(output, target):
  return 10 * torch.log10(1 / F.mse_loss(output, target))


class MS_SSIM(nn.Module):
  def __init__(self, window_size=11, size_average=True, normalize=True):
    super(MS_SSIM, self).__init__()
    self.window_size = window_size
    self.size_average = size_average
    self.normalize = normalize

  def forward(self, output, target):
    return msssim(output, target, window_size=self.window_size, size_average=self.size_average,
                  normalize=self.normalize)


class SSIM(nn.Module):
  def __init__(self, window_size=11, size_average=True):
    super(SSIM, self).__init__()
    self.window_size = window_size
    self.size_average = size_average

  def forward(self, output, target):
    return ssim(output, target, window_size=self.window_size, size_average=self.size_average)


