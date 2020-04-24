import argparse
import time

import torch
import os
from DBLLNet.Model import Lr_Splat,Lr_LocalFeatures,Lr_GlobalFeatures,Guide_PointwiseNN, LinearPredict_BGrid, Fusion, SliceNApply, Net
from DBLLNet.Dataset import LowLightDataSet
from DBLLNet.Tools import compute_PSNR, showIMG
from pytorch_msssim import ssim, msssim

import cv2

from statistics import mean

FPS = []
frameOutputCount = 0
start_time = 0

def testModel():
    print("Splat")
    X = torch.rand(1, 3, 256, 256)
    lrsplat = Lr_Splat(cm=1, sb=16, lb=8, bn=True, lris=256)
    out = lrsplat(X)
    print(list(out.size()))
    print("Local Features")
    X = torch.rand(1, 64, 16, 16)
    lrlocalfeatures = Lr_LocalFeatures(cm=1, sb=16, lb=8, bn=True, lris=256)
    out = lrlocalfeatures(X)
    print(list(out.size()))
    print("Global Features")
    X = torch.rand(2, 64, 16, 16)
    lrglobalfeatures = Lr_GlobalFeatures(cm=1, sb=16, lb=8, bn=True, lris=256)
    out = lrglobalfeatures(X)
    print(list(out.size()))
    print("Fusion and Bilterial Grid")
    LrLocal = torch.rand(2, 64, 16, 16)
    LrGlobal = torch.rand(2, 64)
    fusion = Fusion()
    bg = LinearPredict_BGrid(cm=1, lb=8)
    out = fusion(LrLocal, LrGlobal)
    bg_out = bg(out)
    print(list(bg_out.size()))
    print("GuideNN")
    X = torch.rand(2, 3, 1080, 1920)
    guide = Guide_PointwiseNN(bn=True)
    g_out = guide(X)
    print(list(g_out.size()))
    print("Slice and Apply Coefficients")
    slice_op = SliceNApply()
    bg_out = bg_out.cuda()
    g_out = g_out.cuda()
    X = X.cuda()
    out = slice_op(bg_out, g_out, X)
    print(list(out.size()))
    print("Net")
    lr = torch.rand(2, 3, 256, 256).cuda()
    fr = torch.rand(2, 3, 1080, 1920).cuda()
    net = Net(_cm=1, _sb=16, _lb=8, _bn=True, _lris=256).cuda()
    out = net(lr, fr)[1]
    print(out.shape)

def caculateFPS():
    global frameOutputCount
    global start_time
    global FPS
    if (time.time() - start_time) > 1:
        rt_fps = frameOutputCount / (time.time() - start_time)
        FPS.append(rt_fps)
        print("\nDBLLNet_FPS", rt_fps)
        frameOutputCount = 0
        start_time = time.time()
    frameOutputCount += 1

def test(isEval=True, sb=16, lb=8, cm=1, lris=256, bn=True,
         checkptfolder='/content/gdrive/My Drive/Checkpoint/', checkptname='checkpt.pth',
         testing_path="/content/gdrive/My Drive/Data/RetinexNetData/LOLdataset/eval15"):
    count = 0
    totalmsssimloss = 0
    totalssimloss = 0
    totalPSNR = 0
    totalL1Loss = 0

    CUDA = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    if CUDA:
        dl_pin_memory = True
        device = torch.device("cuda")
    else:
        dl_pin_memory = False
        device = torch.device("cpu")
        return
    print("Cuda", dl_pin_memory)

    testing_dataset = LowLightDataSet(testing_path)
    test_loader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=1, shuffle=True,
                                              pin_memory=dl_pin_memory)

    model = Net(_cm=cm, _sb=sb, _lb=lb, _bn=bn, _lris=lris)

    if os.path.isfile(checkptfolder + checkptname):
        print("=> Loading checkpoint'{}".format(checkptfolder + checkptname))
        checkpt = torch.load(checkptfolder + checkptname)
        model.load_state_dict(checkpt['model_state_dict'])
        print("=> loaded checkpoint '{}' ".format(checkptfolder + checkptname))
    else:
        print("=> no checkpoint found at '{}'".format(checkptfolder + checkptname))
        return

    model.eval()
    model.to(device)
    cv2.namedWindow("DBLLNet Testing", cv2.WND_PROP_FULLSCREEN)

    for batch_idx, (fr, lr, target, target_map) in enumerate(test_loader):

        lr = lr.to(device)
        fr = fr.to(device)
        target = target.to(device)
        g_out, output = model(lr, fr)
        # output = amplification(output)
        l1_criterion = torch.nn.L1Loss()
        l1loss = l1_criterion(output, target)
        print("Image: {}, SSIM: {}, MS-SSIM: {}, L1 Loss {}, PSNR: {} dB ".format(batch_idx,
                                                                                  ssim(output, target,
                                                                                       size_average=True),
                                                                                  msssim(output, target,
                                                                                         size_average=True),
                                                                                  l1loss.item(),
                                                                                  compute_PSNR(output, target)))
        count += 1
        totalssimloss += ssim(output, target, size_average=True)
        totalmsssimloss += msssim(output, target, size_average=True)
        totalPSNR += compute_PSNR(output, target)
        totalL1Loss += l1loss.item()
        imgs_output = []
        imgs_output.append(output.squeeze(0))

        if isEval:
            isShowIMG = False
            isShowGuideMap = False
        else:
            isShowIMG = True
            isShowGuideMap = True

        if isShowIMG:
            if isShowGuideMap:
                imgs_output.append(g_out)
            showIMG(imgs_output)

        if isEval:
            caculateFPS()
            cv_frame = cv2.cvtColor(output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(),cv2.COLOR_RGB2BGR)
            cv2.imshow("DBLLNet Testing", cv_frame)
            cv2.waitKey(1)

    print("Average: SSIM: {},Average MS-SSIM: {},Average L1 Loss {},Average PSNR: {} dB, Average FPS: {}  ".format(
                                                                            totalssimloss / count,
                                                                            totalmsssimloss / count,
                                                                            totalL1Loss / count,

                                                                            totalPSNR / count,
                                                                            mean(FPS)))
    cv2.destroyWindow("DBLLNet Testing")
parser = argparse.ArgumentParser(description='Training')
#Training parameters
parser.add_argument('--data', '-d', type=str,help='testing dataset path',default=None)
parser.add_argument('--checkpoint_folder', '-cpf', type=str, dest='checkpoint_folder', help="Model Checkpoint Folder")
parser.add_argument('--checkpoint', '-cp', type=str, dest='checkpoint', help="Model Checkpoint")
#Network parameters
parser.add_argument('--spatial_bins', '-sb',type=int, dest='spatial_bins',help="Spatial size of the bilteral grid (Size of the spatial BGU bins (pixels)", default=16)
parser.add_argument('--luma_bins', '-lb', type=int,dest='luma_bins',help="Number of channels of the bilteral grid (Number of BGU bins for the luminance)", default=8)
parser.add_argument('--channel_multiplier', '-cm', type=int,dest='channel_multiplier',help="(Factor to control net throughput (number of intermediate channels))", default=1)
parser.add_argument('--low_res_input_size', '-lris', type=int, dest='low_res_input_size', help="input size of the low res stream", default=256)
parser.add_argument('--eval', '-e', type=bool,help='Eval',default=True)

args = parser.parse_args()

if __name__ == '__main__':
    test(isEval=args.eval, sb=args.spatial_bins, lb=args.luma_bins, cm=args.channel_multiplier, lris=args.low_res_input_size, bn=True,
         checkptfolder=args.checkpoint_folder, checkptname=args.checkpoint,
         testing_path=args.data)
