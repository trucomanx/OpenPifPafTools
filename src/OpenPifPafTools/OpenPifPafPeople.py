#!/usr/bin/python


#import openpifpaf
import OpenPifPafTools.OpenPifPafAnnotations as oppa
import OpenPifPafTools.OpenPifPafGetData as oppd


def get_pil_images_with_people(filepath):
    '''
    Lee un archivo de imagen `filepath` y retorna una lista de pil images con una persona por imagen.
    
    :param filepath: Archivo de imagen a analizar.
    :type filepath: str
    :return: Retorna una lista de pil images con una persona por imagen.
    :rtype: list[PIL.Image.Image]
    '''
    annotation,pil_im=oppa.get_openpifpaf_annotation_from_imgpath(filepath);
    
    pil_list=[];
    for annot in annotation:
        tupla=oppd.get_body_bounding_rectangle(annot.data, factor=1.4);
        if not((tupla[0]==0)and(tupla[1]==0)and(tupla[2]==0)and(tupla[3]==0)):
            tupla=oppd.get_valid_bounding_rectangle(tupla, (pil_im.size[0],pil_im.size[1]))
            pil_im_crop = pil_im.crop(tupla);
            pil_list.append(pil_im_crop);
    
    return pil_list;
