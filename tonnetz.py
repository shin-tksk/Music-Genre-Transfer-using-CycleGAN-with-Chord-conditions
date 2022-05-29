import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import distance
from utils import *
from chord import *
import sys

args = sys.argv
GENRU = args[1]
MODE = args[2]

#your directory
ROOT_PATH = '/'

root_name = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B",]

#chord vector(12 dimensions)
a = np.ones(12)
major = math_tonnetz(regular(major_chord(np.diag(a))))
minor = math_tonnetz(regular(minor_chord(np.diag(a))))

#transformation matrix
transform = []
for n in range(12):
    transform.append([math.sin(math.radians(n*210)),math.cos(math.radians(n*210)),
    math.sin(math.radians(n*270)),math.cos(math.radians(n*270)),
    math.sin(math.radians(n*120)),math.cos(math.radians(n*120)),])
transform = np.array(transform).reshape(12,6) 

def chord_detection(tonnetz,root):
    dis = 1.0
    name = "None"
    num = np.zeros(2)
    chord = np.zeros(13)
    for i in range(12):
        dis_major = distance.euclidean(major[i],tonnetz)
        dis_minor = distance.euclidean(minor[i],tonnetz)
        if(dis_major<=dis):
            dis = dis_major
            name = "{}".format(root_name[i])
            num = (i,1)
            chord = major_chord(np.diag(a))[i]
        if(dis_minor<=dis):
            dis = dis_minor
            name = "{}m".format(root_name[i])
            num = (i,2)
            chord = minor_chord(np.diag(a))[i]
    return name,chord,dis,num

def get_key(M,m):
    key = [0,0]
    border = 0 
    M = M/np.sum(M)
    m = m/np.sum(m)
    for i in range(12):
        sum = M[i]+M[(i+5)%12]+M[(i+7)%12]+m[(i+2)%12]+m[(i+4)%12]+m[(i+9)%12]
        if sum > border:
            border = sum
            num = i
    key[0] = num
    return key

def detect_chord_key():
    l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_npy'.format(GENRU,GENRU,MODE)))]
    key_list = np.zeros(12)
    for n in l:
        try:
            print(n)
            SIZE = 1/2
            BASE = 7
            ori = np.load(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_npy'.format(GENRU,GENRU,MODE), n))
            ori = ori.reshape(-1,192,84)
            p = np.delete(ori,slice(BASE*12,84),2)
            p = p.reshape(-1,int(48*SIZE),BASE*12)
            p = np.sum(p,axis=1)
            q = p.reshape(-1,BASE,12)
            sum = np.sum(q,axis=1)

            tonnetz_list = []
            chord_list = []
            major_list = np.zeros(12)
            minor_list = np.zeros(12)

            for i in range(len(sum)):
                root = 13
                for x in range(BASE*12):
                    if(p[i][x]!=0):
                        root = x%12
                        break
                sum[i] = sum[i].reshape(1,12)
                norm = np.sum(sum[i])
                if norm != 0:
                    tonnetz = sum[i]/norm
                else: tonnetz = sum[i]
                tonnetz = np.dot(tonnetz,transform)
                tonnetz = tonnetz.reshape(-1)
                tonnetz_list.append(tonnetz)
                ans = chord_detection(tonnetz,root)
                if ans[3][1] == 1:
                    major_list[ans[3][0]] += 1
                if ans[3][1] == 2:
                    minor_list[ans[3][0]] += 1
                d = ans[1] 
                for _ in range(int(48*SIZE)):
                    chord_list.append(d)
            
            chord = np.array(chord_list).reshape(-1,192,12)
            data = np.concatenate([ori,chord],2) > 0
            data = data.astype(np.int)
            data = data.reshape(-1,192,96,1)
            np.save(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE), os.path.splitext(n)[0] + '.npy'), data)

            #key detection
            M = major_list
            m = minor_list
            key = get_key(M,m)
            key_list[key[0]] += 1

        except:
            print("Wrong:",n)
            continue