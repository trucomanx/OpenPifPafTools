#!/usr/bin/python


import openpifpaf
import PIL
import numpy as np
#import torch
import cv2


def get_openpifpaf_annotation_from_imgpath(filepath):
    '''
    Lee un archivo de imagen `filepath` y retorna las anotaciones y la imagen leida.
    
    :param filepath: Archivo de imagen a analizar.
    :type filepath: str
    :return: Retorna anotaciones y la imagen leida.
    :rtype: list[openpifpaf.annotation.Annotation], PIL.Image.Image
    '''
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
    '''
    Redimenciona una imagen y sus anotaciones.
    
    :param annotation: Anotaciones
    :type annotation: list[openpifpaf.annotation.Annotation]
    :param pil_img: Imagen a ser resdimencionada.
    :type pil_img: PIL.Image.Image
    :param width: Tamaño nuevo de la imagen.
    :type width: int
    :return: Retorna la anotación e imagen rescalada.
    :rtype: list[openpifpaf.annotation.Annotation], PIL.Image.Image
    '''
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

def save_openpifpaf_annotation_in_img(pil_im,annotation,output_file,width=None):
    '''
    A partir de una lista de anotaciones y una imagen 
    crea una imagen en output_file
    
    :param pil_im: Imagen a ser anotada 
    :type pil_im: PIL.Image.Image
    :param annotation: Anotaciones
    :type annotation: list[openpifpaf.annotation.Annotation]
    :param output_file: Path del archivod e salida.
    :type output_file: str
    :param width: Tamaño nuevo de la imagen.
    :type width: int
    
    '''
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


def save_openpifpaf_annotation_in_imgpath(filepath,annotation,output_file):
    '''
    A partir de una lista de anotaciones y un archivo de origen
    crea una imagen en output_file
    
    :param filepath: Path del archivo de entrada a ser anotado 
    :type filepath: str
    :param annotation: Anotaciones
    :type annotation: list[openpifpaf.annotation.Annotation]
    :param output_file: Path del archivod e salida.
    :type output_file: str
    '''
    pil_im = PIL.Image.open(filepath).convert('RGB')
    
    save_openpifpaf_annotation_in_img(pil_im,annotation,output_file)




