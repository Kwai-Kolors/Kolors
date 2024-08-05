# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    # canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    
    def getres(self, oriImg):
        out_res = {}
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            out_res['candidate']=candidate
            out_res['subset']=subset
            out_res['width']=W
            out_res['height']=H
            return out_res

    def __call__(self, oriImg):

        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            _candidate, _subset = self.pose_estimation(oriImg)
            
            subset = _subset.copy()
            candidate = _candidate.copy()
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            out_res = {}
            out_res['candidate']=candidate
            out_res['subset']=subset
            out_res['width']=W
            out_res['height']=H

            return out_res,draw_pose(pose, H, W)
