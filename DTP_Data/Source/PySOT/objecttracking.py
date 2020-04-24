from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch

from PySOT.pysot.core.config import cfg
from PySOT.pysot.models.model_builder import ModelBuilder
from PySOT.pysot.tracker.tracker_builder import build_tracker

def objtracking_init(checkpt, config, cuda, device):
    cfg.merge_from_file(config)
    cfg.CUDA = cuda and cfg.CUDA
    model = ModelBuilder()
    model.load_state_dict(torch.load(checkpt, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)
    return tracker

def track(tracker, frame):
    outputs = tracker.track(frame)
    bbox = list(map(int, outputs['bbox']))

    cv2.rectangle(frame, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (0, 255, 0), 3)
    return str(bbox), frame
