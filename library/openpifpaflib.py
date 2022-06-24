#!/usr/bin/python


import openpifpaf
import PIL
import numpy as np
import torch
import cv2


'''
Retorna: 
* Una lista de anotaciones con todas las personas en una foto (filepath).
Cada anotacion annot tiene: {annot.skeleton, annot.keypoints, annot.data}
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
* Una matriz 3D numpy que representa los pixels en la imagen 
'''
def get_openpifpaf_annotation_from_imgpath(filepath):
    
    pil_im = PIL.Image.open(filepath).convert('RGB')
    
    '''
    im = np.asarray(pil_im)
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
    return annots, pil_im

def resize_openpifpaf_annotation_and_pil_img(annotation,pil_img, width=None):
    if (width==None):
        return annotation,pil_img
    
    width=int(width);
    factor=width/pil_img.size[0];
    height=int(pil_img.size[1]*factor);
    
    pil_img = pil_img.resize((width, height));
    
    N=len(annotation);
    for n in range(N):
        annotation[n].data[:,0]=annotation[n].data[:,0]*factor;
        annotation[n].data[:,1]=annotation[n].data[:,1]*factor;
    
    return annotation,pil_img
'''
A partir de una lista de anotaciones y una imagen im (numpy)
crea una imagen en output_file
'''
def save_openpifpaf_annotation_in_img(pil_im,annotation,output_file,width=None):
    
    annotation,pil_im=resize_openpifpaf_annotation_and_pil_img(annotation,pil_im,width);
    im=np.asarray(pil_im);
    
    #keypoint_painter = openpifpaf.show.KeypointPainter()
    #annotation_painter = openpifpaf.show.AnnotationPainter(painters={"Annotation": keypoint_painter})
    annotation_painter = openpifpaf.show.AnnotationPainter()
    
    '''
    N=len(annotation);
    subtext=[];
    for n in range(N):
        subtext.append(str(n));
    '''
    subtext=None;
    with openpifpaf.show.image_canvas(im,output_file) as ax:
        annotation_painter.annotations(ax, annotation,subtexts=subtext)

'''
A partir de una lista de anotaciones y un archivo de origen
crea una imagen en output_file
'''
def save_openpifpaf_annotation_in_imgpath(filepath,annotation,output_file):
    pil_im = PIL.Image.open(filepath).convert('RGB')
    
    save_openpifpaf_annotation_in_img(pil_im,annotation,output_file)


################################################################################
'''
retorna la distancia entre los ojo
'''
def get_between_eyes_distance(annotation_data):
    eyel1=annotation_data[1,0];
    eyel2=annotation_data[1,1];
    
    eyer1=annotation_data[2,0];
    eyer2=annotation_data[2,1];
    
    if((eyel1<=0) or (eyel2<=0)):
        return 0.0;
    
    if((eyer1<=0) or (eyer2<=0)):
        return 0.0;
    return np.sqrt((eyel1-eyer1)**2+(eyel2-eyer2)**2)
    
'''
retorna la distancia media entre un ojo y la nariz
'''
def get_mean_nose_eye_distance(annotation_data):
    nose1=annotation_data[0,0];
    nose2=annotation_data[0,1];
    
    eyel1=annotation_data[1,0];
    eyel2=annotation_data[1,1];
    
    eyer1=annotation_data[2,0];
    eyer2=annotation_data[2,1];
    
    if((nose1<=0) or (nose2<=0)):
        return 0.0;
    dl=0;
    if((eyel1>0) or (eyel2>0)):
        dl=np.sqrt((eyel1-nose1)**2+(eyel2-nose2)**2);
    
    dr=0;
    if((eyer1>0) or (eyer2>0)):
        dr=np.sqrt((eyer1-nose1)**2+(eyer2-nose2)**2);
        
    if(dl<=0):
        return dr;
    
    if(dr<=0):
        return dl;
    
    return (dl+dr)/2.0;
