#!/usr/bin/python

import openpifpaf
import numpy as np

import OpenPifPafTools.OpenPifPafGetData as oppgd

def standardize_matrix(matrix):
    '''
    Se existe uma linha negativa entao todo a linha e' zero
    Se a probabilidade e' menor igual a cero a linha e' zero
    '''
    # Cria uma máscara booleana para as linhas onde algum elemento é negativo
    mask_negative = (matrix < 0).any(axis=1)
    
    # Cria uma máscara booleana para as linhas onde o elemento da terceira coluna é menor ou igual a zero
    mask_third_column = matrix[:, 2] <= 0
    
    # Combina as duas máscaras usando a operação lógica OR
    mask = mask_negative | mask_third_column
    
    # Atribui zero a todas as linhas que atendem à condição
    matrix[mask] = 0
    
    return matrix
    
class Detector:
    def __init__(self, checkpoint='shufflenetv2k16', body_factor=1.0, face_factor=1.0):

        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint);
        self.body_factor = body_factor;
        self.face_factor = face_factor;

    def process_image(self,pil_image):
        skel_vec, body_roi, face_roi, body_bbox, face_bbox =self.process_image_full(pil_image);
        return skel_vec, body_roi, face_roi;
        
    def process_image_full(self,pil_image):
        """
        Se nao se achou pessoas entao todos seus elementos sao None.
        Se se achou uma pessoa, entao a  saida  e' trabalhavel.
        Se face nao e' achado se retorna None para face e bouding box.
        Se body nao e' achado se retorna None para body e bouding box.
        """
        
        annotation1, _, _ = self.predictor.pil_image(pil_image);
        
        if not annotation1:
            return None, None, None, None, None;

        # Extract keypoints
        for annot in annotation1: 
            annot_data=standardize_matrix(annot.data);
            
            # face
            (xi,yi,xo,yo)=oppgd.get_face_bounding_rectangle(annot_data,factor=self.face_factor);
            xi=int(xi);        yi=int(yi);
            xo=int(xo);        yo=int(yo);
            
            if (xi,yi,xo,yo)==(0,0,0,0):
                face_bbox=None;
                face_roi=None;
            else:
                face_bbox=(xi,yi,xo,yo);
                face_roi=pil_image.crop(face_bbox);
            
            # body
            (xi,yi,xo,yo)=oppgd.get_body_bounding_rectangle(annot_data,factor=self.body_factor);
            xi=int(xi);        yi=int(yi);
            xo=int(xo);        yo=int(yo);
            
            if (xi,yi,xo,yo)==(0,0,0,0):
                body_bbox=None;
                body_roi=None;
            else:
                body_bbox=(xi,yi,xo,yo);
                body_roi=pil_image.crop(body_bbox);
            
            mat=self.zero_out_rows(annot_data);
            
            # skel
            skel_vec=mat.reshape((-1,));
            
            
        return skel_vec, body_roi, face_roi, body_bbox, face_bbox;
    
    def process_image_list(self,pil_image_list):
        skel_vec_list, body_roi_list, face_roi_list, body_bbox_list, face_bbox_list = self.process_image_full_list(pil_image_list);
        
        return skel_vec_list, body_roi_list, face_roi_list;
        
    def process_image_full_list(self,pil_image_list):
        skel_vec_list=[];
        body_roi_list=[];
        face_roi_list=[];
        body_bbox_list=[];
        face_bbox_list=[];
        
        for pil_image in pil_image_list:
            skel_vec, body_roi, face_roi, body_bbox, face_bbox = self.process_image_full(pil_image);
            
            skel_vec_list.append(skel_vec);
            body_roi_list.append(body_roi);
            face_roi_list.append(face_roi);
            body_bbox_list.append(body_bbox);
            face_bbox_list.append(face_bbox);
            
        return skel_vec_list, body_roi_list, face_roi_list, body_bbox_list, face_bbox_list;
    
    def zero_out_rows(self,matrix):
        """
        Zeros out rows in the matrix if any element in the row is less than 0.
        
        Args:
        matrix (np.ndarray): A numpy array with shape (n, 3).
        
        Returns:
        np.ndarray: The modified matrix.
        """
        # Ensure the input is a numpy array
        #matrix = np.array(matrix)
        
        # Check if the matrix has 3 columns
        if matrix.shape[1] != 3:
            raise ValueError("The input matrix must have exactly 3 columns.")
        
        # Find rows where any element is less than or equal to z
        rows_to_zero = np.any(matrix < 0.0, axis=1)
        
        
        # Set those rows to zero
        matrix[rows_to_zero] = 0.0;
        
        return matrix

