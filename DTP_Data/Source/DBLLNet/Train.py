import argparse
import os
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from DBLLNet.Dataset import LowLightDataSet
from DBLLNet.Model import Net

from DBLLNet.Tools import MS_SSIM, SSIM,compute_PSNR
from pytorch_msssim import ssim, msssim

#FOR DEBUG
# training_data_dir = 'C:\\Users\\Family\\Documents\\FYP_CV_Code\\RetinexNetData\\LOLdataset\\our485'
#
parser = argparse.ArgumentParser(description='Training')
#Training parameters
parser.add_argument('--data', '-d', type=str,help='dataset path',default=None)
parser.add_argument('--checkpoint_folder', '-cpf', type=str, dest='checkpoint_folder', help="Model Checkpoint Folder")
parser.add_argument('--checkpoint', '-cp', type=str, dest='checkpoint', help="Model Checkpoint")
parser.add_argument('--batch_size', '-bs', type=int,help='Batch Size',dest="batch_size",default=16)
parser.add_argument('--batchnorm', '-bn', dest='batch_norm',help="Enable Batch Norm", action='store_true',default=True)
#Network parameters
parser.add_argument('--spatial_bins', '-sb',type=int, dest='spatial_bins',help="Spatial size of the bilteral grid (Size of the spatial BGU bins (pixels)", default=16)
parser.add_argument('--luma_bins', '-lb', type=int,dest='luma_bins',help="Number of channels of the bilteral grid (Number of BGU bins for the luminance)", default=8)
parser.add_argument('--channel_multiplier', '-cm', type=int,dest='channel_multiplier',help="(Factor to control net throughput (number of intermediate channels))", default=1)
parser.add_argument('--low_res_input_size', '-lris', type=int, dest='low_res_input_size', help="input size of the low res stream", default=256)
parser.add_argument('--epoch', '-e', type=int, dest='epoch', help="number of epoch", default=50)
parser.add_argument('--weight_decay', '-wd', type=float, dest='weight_decay', help="Weight Decay", default=10 ** -8)
parser.add_argument('--learning_rate', '-lr', type=float, dest='learning_rate', help="Learning Rate", default=10 ** -4)
parser.add_argument('--isResume', '-r', type=bool, dest='isResume', help="Resume Training", default=False)

args = parser.parse_args()


def train(sb, lb, cm, lris, epochs, w_decay, bs, bn, learning_rate, isResume=True,
          checkptfolder='/content/gdrive/My Drive/Checkpoint/1/',
          training_path="/content/gdrive/My Drive/Data/RetinexNetData/BrighteningTrain/", checkptname='checkpt.pth',
          checkptinterval=10):
    CUDA = torch.cuda.is_available()
    torch.cuda.empty_cache()
    if CUDA:
        dl_pin_memory = True
        device = torch.device("cuda")
    else:
        dl_pin_memory = False
        device = torch.device("cpu")
        return
    print("Cuda", dl_pin_memory)

    training_dataset = LowLightDataSet(training_path)
    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=bs, shuffle=True,
                                               pin_memory=dl_pin_memory)

    model = Net(_cm=cm, _sb=sb, _lb=lb, _bn=bn, _lris=lris)
    msssim_criterion = MS_SSIM()
    ssim_criterion = SSIM()
    mse_criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    curr_epoch = 0
    losslogger = []

    if isResume:
        if os.path.isfile(checkptfolder + checkptname):
            print("=> Loading checkpoint'{}".format(checkptfolder + checkptname))
            checkpt = torch.load(checkptfolder + checkptname)
            curr_epoch = checkpt['epoch']
            epochs -= curr_epoch
            model.load_state_dict(checkpt['model_state_dict'])
            optimizer.load_state_dict(checkpt['optimizer_state_dict'])
            losslogger = checkpt['losslogger']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            print(
                "=> loaded checkpoint '{}' (start from epoch {})".format(checkptfolder + checkptname, checkpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkptfolder + checkptname))
            return
    else:
        print("=> No checkpoint will be used")

    model.to(device)

    for e in range(epochs):
        e_losslogger = []
        model.train()

        for batch_idx, (fr, lr, target, target_map) in enumerate(train_loader):
            optimizer.zero_grad()
            lr = lr.to(device)
            fr = fr.to(device)
            target_map = target_map.to(device).squeeze(1)
            target = target.to(device)
            g_out, output = model(lr, fr)
            msssimloss = 1 - msssim_criterion(output, target)
            ssimloss = 1 - ssim_criterion(output, target)
            l1loss = l1_criterion(output, target)
            l2loss = mse_criterion(output, target)
            loss = msssimloss
            e_losslogger.append(loss)
            loss.backward()
            print(
                "Epoch: {}, Batch: {},  SSIM: {}, MS-SSIM: {}, L1 Loss: {}, L2 Loss:{}, PSNR: {} dB".format(curr_epoch,
                                                                                                            batch_idx,
                                                                                                            ssim(output,
                                                                                                                 target,
                                                                                                                 size_average=True),
                                                                                                            msssim(
                                                                                                                output,
                                                                                                                target,
                                                                                                                size_average=True),
                                                                                                            l1loss.item(),
                                                                                                            l2loss.item(),
                                                                                                            compute_PSNR(
                                                                                                                output,
                                                                                                                target)))
            optimizer.step()

        losslogger.append(e_losslogger)
        if (curr_epoch + 1) % checkptinterval == 0:
            saveCheckpt(checkptfolder, curr_epoch, model, optimizer, losslogger)
        curr_epoch += 1


def saveCheckpt(checkptfolder, curr_epoch, model, optimizer, losslogger, batch_idx='_'):
    checkptname = 'checkpt_epoch_{}_batch_{}.pth'.format(curr_epoch, batch_idx)
    print("Saving checkpoint {}".format(checkptname))
    torch.save({
        'epoch': curr_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losslogger': losslogger,
    }, checkptfolder + checkptname)
    print("{} is saved".format(checkptname))

if __name__ == '__main__':
    train(sb=args.spatial_bins, lb=args.luma_bins, cm=args.channel_multiplier, lris=args.low_res_input_size, epochs=args.epoch, w_decay=args.weight_decay, bs=args.batch_size, bn=args.batch_norm, learning_rate=args.learning_rate,isResume=False,training_path=args.data, checkptfolder=args.checkpoint_folder)
#Python train.py --data C:\Users\Family\Documents\FYP_CV_Code\hdrnet-related\RetinexNetData\Combined\Train1485 --checkpoint_folder checkpoint\