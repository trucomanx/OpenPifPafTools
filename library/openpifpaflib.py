#!/usr/bin/python


import openpifpaf
import PIL
import numpy as np
import torch
import cv2


'''
Retorna un lista de anotaciones con todas las personas en una foto (filepath).
Cada anotacion annot tiene: annot.skeleton,annot.keypoints, annot.data
Cada dato de anotacion annot.data tiene este formato:
[[377.11533   268.32425     0.9934822]
 [393.13257   250.68027     0.9986747]
 [363.1835    250.94507     0.9893863]
 [417.96884   258.2936      0.9928429]
 [345.3207    259.28903     0.9730895]
 [455.51154   356.56622     0.9805316]
 [300.9324    342.88437     0.9678486]
 [550.05554   413.7901      0.9721895]
 [188.23953   365.4678      0.9518851]
 [567.95154   356.78802     0.9664481]
 [172.63026   289.0084      0.9349717]
 [383.33032   486.02448     0.8884974]
 [282.2343    503.27646     0.7614233]
 [  0.         -7.          0.       ]
 [  0.         -7.          0.       ]
 [  0.         -7.          0.       ]
 [  0.         -7.          0.       ]]
y esta orden (annot.keypoints):
['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
'''
def get_openpifpaf_annotation_from_imgpath(filepath):
    
    pil_im = PIL.Image.open(filepath).convert('RGB')
    im = np.asarray(pil_im)
    '''
    with openpifpaf.show.image_canvas(im) as ax:
        pass
    cv2.imshow('image',im)
    cv2.waitKey(0)
    '''

    
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')#'shufflenetv2k16-wholebody'
    annots, gt_anns, image_meta = predictor.pil_image(pil_im);
    
    '''
    for annot in annots: # annots[1]
        print(" ")
        print(annot.skeleton)
        print(annot.keypoints)
        print(annot.data)
    '''
    #print(gt_anns) # []
    #print(image_meta) # {'dataset_index': 0, 'offset': array([ 0., -2.]), 'scale': array([1., 1.]), 'rotation': {'angle': 0.0, 'width': None, 'height': None}, 'valid_area': array([  0.,   2., 799., 668.]), 'hflip': False, 'width_height': array([800, 669])}
    return annots
 
def save_openpifpaf_annotation_in_imgpath(filepath,annotation,output_file):
    pil_im = PIL.Image.open(filepath).convert('RGB')
    im = np.asarray(pil_im)
    annotation_painter = openpifpaf.show.AnnotationPainter()
    
    with openpifpaf.show.image_canvas(im,output_file) as ax:
        annotation_painter.annotations(ax, annotation)
        
