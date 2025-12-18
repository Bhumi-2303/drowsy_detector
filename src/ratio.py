import numpy as np 
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear =  (A + B) / (2.0 * C)
    return ear
    
from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # vertical distances
    A = dist.euclidean(mouth[3], mouth[9])   # 51-57
    B = dist.euclidean(mouth[2], mouth[10])  # 50-58
    C = dist.euclidean(mouth[4], mouth[8])   # 52-56
    
    # horizontal distance
    D = dist.euclidean(mouth[0], mouth[6])   # 48-54

    mar = (A + B + C) / (2.0 * D)
    return mar


