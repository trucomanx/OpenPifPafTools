#!/usr/bin/python

import openpifpaf
import numpy as np
from numpy import linalg as LA


################################################################################

def valid_cord(v, p=1.0):
    return valid_cordxy(v[0],v[1], p=p);

def valid_cordxy(a, b, p=1.0):
    if p<=0:
        return False;
    
    # Verifica se ambas as variáveis são não negativas
    if a < 0 or b < 0:
        return False;
    
    # Verifica se ambas as variáveis não são iguais a zero
    if a == 0 and b == 0:
        return False;
    
    # Caso as condições anteriores sejam atendidas, a validação é bem-sucedida
    return True;

################################################################################

def get_between_eyes_distance(annotation_data):
    '''
    Retorna la distancia entre los ojos (pixels).
    
    .. warning::
        En caso de que no sea possible el cálculo se retorna 0.
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: Retorna la distancia entre los ojos (pixels).
    :rtype: float
    '''
    eyel1=annotation_data[1,0];
    eyel2=annotation_data[1,1];
    
    eyer1=annotation_data[2,0];
    eyer2=annotation_data[2,1];
    
    if((eyel1<=0) or (eyel2<=0)):
        return 0.0;
    
    if((eyer1<=0) or (eyer2<=0)):
        return 0.0;
    return np.sqrt((eyel1-eyer1)**2+(eyel2-eyer2)**2)
    

def get_mean_nose_eye_distance(annotation_data):
    '''
    Retorna la distancia media entre un ojo y la nariz (pixels).
    
    .. warning::
        En caso de que no sea possible el cálculo se retorna 0.
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :return: Retorna la distancia media entre un ojo y la nariz (pixels).
    :rtype: float
    '''
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


################################################################################

def get_between_ears_distance(annotation_data):
    '''
    Retorna la distancia entre las orejas (pixels).
    
    .. warning::
        En caso de que no sea possible el cálculo se retorna 0.
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: Retorna la distancia entre las orejas (pixels).
    :rtype: float
    '''
    earl1=annotation_data[3,0];
    earl2=annotation_data[3,1];
    
    earr1=annotation_data[4,0];
    earr2=annotation_data[4,1];
    
    if((earl1<=0) or (earl2<=0)):
        return 0.0;
    
    if((earr1<=0) or (earr2<=0)):
        return 0.0;
    return np.sqrt((earl1-earr1)**2+(earl2-earr2)**2)

################################################################################

def get_between_ear_eye_distance(annotation_data):
    '''
    Retorna la distancia media entre la oreja y un ojo (pixels).
    
    .. warning::
        En caso de que no sea possible el cálculo se retorna 0.
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: Retorna la distancia entre las orejas (pixels).
    :rtype: float
    '''
    eyel=np.array([annotation_data[1,0], annotation_data[1,1]]);
    eyer=np.array([annotation_data[2,0], annotation_data[2,1]]);
    
    earl=np.array([annotation_data[3,0], annotation_data[3,1]]);
    earr=np.array([annotation_data[4,0], annotation_data[4,1]]);
    
    d=0;
    n=0;
    if((earl[0]>0)and(earl[1]>0)) and ((eyel[0]>0)and(eyel[1]>0)):
        d=d+LA.norm(earl-eyel);
        n=n+1;
    if((earr[0]>0)and(earr[1]>0)) and ((eyer[0]>0)and(eyer[1]>0)):
        d=d+LA.norm(earr-eyer);
        n=n+1;
    if(n==0):
        return 0;
    d=d/n;
    return d;
    
################################################################################

def get_between_shoulders_distance(annotation_data):
    '''
    Retorna la distancia entre los hombros (pixels).
    
    .. warning::
        En caso de que no sea possible el cálculo se retorna 0.
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: Retorna la distancia entre los hombros (pixels).
    :rtype: float
    '''
    shoulderl1=annotation_data[5,0];
    shoulderl2=annotation_data[5,1];
    
    shoulderr1=annotation_data[6,0];
    shoulderr2=annotation_data[6,1];
    
    if((shoulderl1<=0) or (shoulderl2<=0)):
        return 0.0;
    
    if((shoulderr1<=0) or (shoulderr2<=0)):
        return 0.0;
    return np.sqrt((shoulderl1-shoulderr1)**2+(shoulderl2-shoulderr2)**2)
    
################################################################################

def get_face_bounding_rectangle_new1(annotation_data, factor=1.0):
    # Extrair os pontos
    nose = np.array([annotation_data[0, 0], annotation_data[0, 1]])
    
    eyel = np.array([annotation_data[1, 0], annotation_data[1, 1]])
    eyer = np.array([annotation_data[2, 0], annotation_data[2, 1]])
    
    earl = np.array([annotation_data[3, 0], annotation_data[3, 1]])
    earr = np.array([annotation_data[4, 0], annotation_data[4, 1]])
    
    shoulderl = np.array([annotation_data[5, 0], annotation_data[5, 1]])
    shoulderr = np.array([annotation_data[6, 0], annotation_data[6, 1]])

    # Filtrar os pontos válidos
    pontos = [nose, eyel, eyer, earl, earr];
    pontos_validos = np.array([p for p in pontos if valid_cord(p)]);
    
    if pontos_validos.size == 0:
        return (0,0,0,0);  # Nenhum ponto válido
    
    # Calcular o ângulo de inclinação usando os olhos
    delta=np.array([0.0,0.0]);
    n=0;
    if valid_cord(shoulderl) and valid_cord(shoulderr):
        delta += 0.33*(shoulderl - shoulderr);
        n +=1;
    if valid_cord(eyel) and valid_cord(eyer):
        delta += 2.0*(eyel - eyer);
        n +=1;
    if valid_cord(earl) and valid_cord(earr):
        delta += (earl - earr);
        n +=1;
    if valid_cord(earl) and valid_cord(eyel):
        delta += 3.0*(earl-eyel);
        n +=1;
    if valid_cord(eyer) and valid_cord(earr):
        delta += 3.0*(eyer-earr);
        n +=1;
    
    if n>0:
        delta=delta/n;
        angulo = np.arctan2(delta[1], delta[0]);
    else:
        return (0,0,0,0);
    
    
    # Width da face
    width=np.linalg.norm(delta)*factor;
    height=width*1.5*factor;

    # centro
    centro = np.mean(pontos_validos, axis=0);
    #print('centro',centro,width,height)
    
    Pbl=np.array([-0.5*width,-0.5*height]);
    Pbr=np.array([+0.5*width,-0.5*height]);
    Ptl=np.array([-0.5*width,+0.5*height]);
    Ptr=np.array([+0.5*width,+0.5*height]);
    
    # 
    rotacionados = []
    for p in [Pbl, Pbr, Ptl, Ptr]:
        x, y = p
        x_rot = x * np.cos(-angulo) - y * np.sin(-angulo)
        y_rot = x * np.sin(-angulo) + y * np.cos(-angulo)
        rotacionados.append([x_rot, y_rot]+centro)

    # Calcular o bounding box com os pontos rotacionados
    rotacionados = np.array(rotacionados)
    x_min, y_min = np.min(rotacionados, axis=0)
    x_max, y_max = np.max(rotacionados, axis=0)
        
    res=(x_min, y_min, x_max, y_max);

    # Retornar a tupla no formato (xi, yi, xo, yo)
    return res
    

def get_face_bounding_rectangle(annotation_data, factor=1.0):
    '''
    Retorna un bounding box de la cara.
    En caso de error retorna una tupla (0,0,0,0).
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: (left, upper, right, lower)
    :rtype: tupla-4
    '''
    # 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder'

    nose=np.array([annotation_data[0,0], annotation_data[0,1]]);
    
    eyel=np.array([annotation_data[1,0], annotation_data[1,1]]);
    eyer=np.array([annotation_data[2,0], annotation_data[2,1]]);
    
    earl=np.array([annotation_data[3,0], annotation_data[3,1]]);
    earr=np.array([annotation_data[4,0], annotation_data[4,1]]);
    
    shoulderl=np.array([annotation_data[5,0], annotation_data[5,1]]);
    shoulderr=np.array([annotation_data[6,0], annotation_data[6,1]]);
    
    #print('nose',nose)
    #print('eyel',eyel)
    #print('eyer',eyer)
    #print('earl',earl)
    #print('earr',earr)
    #print('shoulderl',shoulderl)
    #print('shoulderr',shoulderr)
    # c0 es el centro de la cara
    c0=np.array([0,0]);
    n=0;
    if valid_cord(nose):
        c0=c0+nose;
        n=n+1;
    if valid_cord(eyel):
        c0=c0+eyel;
        n=n+1;
    if valid_cord(eyer):
        c0=c0+eyer;
        n=n+1;
    if valid_cord(earl):
        c0=c0+earl;
        n=n+1;
    if valid_cord(earr):
        c0=c0+earr;
        n=n+1;
    if(n==0):
        return (0,0,0,0);
    c0=c0/n;
    #print('c0',c0)
    
    #direccion izquierda - derecha
    v=np.array([0,0]);
    n=0;
    if valid_cord(shoulderl) and  valid_cord(shoulderr):
        v=v+0.5*(shoulderl-shoulderr);
        n=n+1;
    if valid_cord(eyel) and valid_cord(eyer):
        v=v+eyel-eyer;
        n=n+1;
    if valid_cord(earl) and valid_cord(earr):
        v=v+earl-earr;
        n=n+1;
    if valid_cord(earl) and valid_cord(eyel):
        v=v+earl-eyel;
        n=n+1;
    if valid_cord(eyer) and valid_cord(earr):
        v=v+eyer-earr;
        n=n+1;
    if(n==0):
        return (0,0,0,0);
    v=v/n;
    
    #calcular width
    deye=get_between_eyes_distance(annotation_data);
    dear=get_between_ears_distance(annotation_data);
    dshoulder=get_between_shoulders_distance(annotation_data);
    deyeear=get_between_ear_eye_distance(annotation_data);
    width=0;
    n=0;
    if(deyeear!=0):
        width=width+3*deyeear;
        n=n+1;
    if(deye!=0):
        width=width+5*deye;
        n=n+1;
    if(dear!=0):
        width=width+dear;
        n=n+1;
    if(dshoulder!=0):
        width=width+0.5*dshoulder;
        n=n+1;
    if(n==0):
        return (0,0,0,0);
    width=width/n;
    
    # calcular height
    deyenose=get_mean_nose_eye_distance(annotation_data);
    height=0;
    if(deyenose!=0):
        height=5.0*deyenose;
    else:
        height=width*2.5;
    
    if((height==0)or(width==0)):
        return (0,0,0,0);
    
    if((v[0]==0)and(v[1]==0)):
        return (0,0,0,0);
    
    v=v/LA.norm(v);
    r=np.array([-v[1],v[0]]);
    
    if factor!=1.0:
        width = width*factor;
        height=height*factor;
    
    p1=c0-v*width/2-r*height/2;
    p2=c0+v*width/2-r*height/2;
    p3=c0+v*width/2+r*height/2;
    p4=c0-v*width/2+r*height/2;
    
    x0=np.array([p1[0],p2[0],p3[0],p4[0]]);
    x1=np.array([p1[1],p2[1],p3[1],p4[1]]);
    
    t0=np.max([0,np.min(x0)]);
    t1=np.max([0,np.min(x1)]);
    return (t0,t1,np.max(x0),np.max(x1))
    
################################################################################

def get_body_bounding_rectangle(annotation_data, factor=1.2):
    '''
    Retorna un bounding box del cuerpo.
    En caso de error retorna una tupla (0,0,0,0).
    
    :param annotation_data: Matriz de 3 columnas y 17  
    :type annotation_data: numpy.ndarray
    :return: (left, upper, right, lower)
    :rtype: tupla-4
    '''
    p=get_face_bounding_rectangle(annotation_data);
    
    pm=np.array([ [p[0], p[1]],[p[2], p[3]] ]);
    b=annotation_data[:,0:2];
    N=b.shape[0];
    for n in range(N):
        if ((b[n,0]>0)and(b[n,1]>0)):
            pm=np.concatenate((pm,[b[n,:]]), axis=0);
    
    N=pm.shape[0];
    mean=np.mean(pm,axis=0);
    if(factor!=1.0):
        for n in range(N):
            pm[n,:]=mean+(pm[n,:]-mean)*factor;
            
    x0=pm[:,0]; x1=pm[:,1];
    
    t0=np.max([0,np.min(x0)]);
    t1=np.max([0,np.min(x1)]);
    return (t0,t1,np.max(x0),np.max(x1))
    
################################################################################

def get_valid_bounding_rectangle(tupla, size):
    '''
    Retorna una tupla-4 válido, menor o igual a size.
    
    :param size: (width, height) 
    :type size: tupla-2
    :param tupla: (left, upper, right, lower)
    :rtype: tupla-4
    '''
    t0=np.max([0,tupla[0]]);
    t1=np.max([0,tupla[1]]);
    t2=np.min([size[0],tupla[2]]);
    t3=np.min([size[1],tupla[3]]);
    return (t0,t1,t2,t3);
