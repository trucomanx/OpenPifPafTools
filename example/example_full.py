#!/usr/bin/python3 

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

import sys
sys.path.append('../src') 

import PIL
import OpenPifPafTools.RoiDetector    as opp


det=opp.Detector(checkpoint='shufflenetv2k16',body_factor=1.0,face_factor=0.9);

pil_img=PIL.Image.open('../dataset/filename463.jpg');

skel_vec_list, body_roi_list, face_roi_list=det.process_image_list([pil_img,pil_img]);

for n in range(len(face_roi_list)):
    body_roi_list[n].save("body_roi_"+str(n)+".png");
    face_roi_list[n].save("face_roi_"+str(n)+".png");
    print(skel_vec_list[n])


