import numpy as np
import math
from utils import *

#transformation matrix
transform = []
root_name = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B",]
for n in range(12):
    transform.append([math.sin(math.radians(n*210)),math.cos(math.radians(n*210)),
    math.sin(math.radians(n*270)),math.cos(math.radians(n*270)),
    math.sin(math.radians(n*120)),math.cos(math.radians(n*120)),])
transform = np.array(transform).reshape(12,6)

a = np.ones(12)
root = np.diag(a)

def regular(c):
    chroma = []
    for i in  range(12):
        norm = np.sum(c[i])
        chroma.append(c[i]/norm)
    return chroma

def major_chord(r):
    chroma = []
    for i in range(12):
        r[i][(i+4)%12] = 1
        r[i][(i+7)%12] = 1
        chroma.append(r[i])
    return chroma

def minor_chord(r):
    chroma = []
    for i in range(12):
        r[i][(i+3)%12] = 1
        r[i][(i+7)%12] = 1
        chroma.append(r[i])
    return chroma

def math_tonnetz(t):
    tonnetz = []
    for i in range(12):
        tonnetz.append(np.dot(t[i].reshape(1,12),transform).reshape(-1))
    return tonnetz