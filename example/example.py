#!/usr/bin/python3 

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

import sys
sys.path.append('../src') 

import PIL
import OpenPifPafTools.RoiDetector    as opp


det=opp.Detector(checkpoint='shufflenetv2k16',body_factor=1.0,face_factor=0.9,face_method=1);

pil_img=PIL.Image.open('../dataset/filename463-rot2.jpg');

skel_vec, body_roi, face_roi=det.process_image(pil_img);


body_roi.save("body_roi.png")
face_roi.save("face_roi.png")
print(skel_vec)


