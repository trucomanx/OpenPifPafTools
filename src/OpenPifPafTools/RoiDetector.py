#!/usr/bin/python

import openpifpaf
import numpy as np

import OpenPifPafTools.OpenPifPafGetData as oppgd


class Detector:
    def __init__(self, checkpoint='shufflenetv2k16'):

        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint);

    def process_image(self,pil_image):
        
        annotation1, _, _ = self.predictor.pil_image(pil_image);
        
        if not annotation1:
            return None, None, None

        # Extract keypoints
        for annot in annotation1: 
            # body
            (xi,yi,xo,yo)=oppgd.get_face_bounding_rectangle(annot.data,factor=1.0);
            xi=int(xi);        yi=int(yi);
            xo=int(xo);        yo=int(yo);
            
            if (xi,yi,xo,yo)==(0,0,0,0):
                face_roi=None;
            else:
                face_roi=pil_image.crop((xi,yi,xo,yo));
            
            # face
            (xi,yi,xo,yo)=oppgd.get_body_bounding_rectangle(annot.data,factor=1.0);
            xi=int(xi);        yi=int(yi);
            xo=int(xo);        yo=int(yo);
            
            if (xi,yi,xo,yo)==(0,0,0,0):
                body_roi=None;
            else:
                body_roi=pil_image.crop((xi,yi,xo,yo));
            
            mat=self.zero_out_rows(annot.data);
            
            # skel
            skel_vec=mat.reshape((-1,));
            
            
        return skel_vec, body_roi, face_roi;
    
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

