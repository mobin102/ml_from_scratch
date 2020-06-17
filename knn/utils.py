import numpy as np 
def eculidean_distance(x1,x2):
    ss = np.sum((x1 - x2)**2) #ss: sum of square
    return np.sqrt(ss)