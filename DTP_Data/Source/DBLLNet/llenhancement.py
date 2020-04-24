import cv2
import torch
from torchvision import transforms
import numpy as np

from DBLLNet.Model import Net

backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=500)

def llenhance_init(checkpoint, device, cm=1, sb=16, lb=8,bn=True,lris=256):
    model = Net(_cm=cm, _sb=sb, _lb=lb, _bn=bn, _lris=lris)
    checkpt = torch.load(checkpoint)
    model.load_state_dict(checkpt['model_state_dict'])
    model.eval().to(device)
    return model

def enhance(model, frame, device,mode="DBLLNet"):
    if mode == "DBLLNet":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lrTransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        frTransform = transforms.Compose([
            transforms.ToTensor()
        ])
        lr_frame = lrTransform(frame)
        fr_frame = frTransform(frame)
        lr = lr_frame.to(device).unsqueeze(0)
        fr =  fr_frame.to(device).unsqueeze(0)
        _, output = model(lr,fr)
        output = convert2CVFormat(output)
    elif mode == "HistogramEqual":
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb_frame)
        cv2.equalizeHist(channels[0],channels[0])
        cv2.merge(channels,ycrcb_frame)
        output = cv2.cvtColor(ycrcb_frame,cv2.COLOR_YCrCb2BGR,None)
    return output

def convert2CVFormat(input):
    input = input.squeeze(0)
    input = input.permute(1,2,0).cpu().detach().numpy()
    cv_input = cv2.cvtColor(input,cv2.COLOR_RGB2BGR)
    cv_input = cv2.normalize(cv_input, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return cv_input

