"""
Some codes from https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer
"""

from random import shuffle
import numpy as np
import os
import shutil
from utils import *

test_ratio = 0.1
#your directory
ROOT_PATH = '/'

def shuffle_data(genru):
     if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_test/origin_midi'.format(genru,genru))):
          os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_test/origin_midi'.format(genru,genru)))
     shutil.copytree(os.path.join(ROOT_PATH, 'MIDI/midi_origin/{}_midi'.format(genru)),
          os.path.join(ROOT_PATH, 'MIDI/{}/{}_train/origin_midi'.format(genru,genru)))

     midi = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/midi_origin/{}_midi'.format(genru)))]
     #print(midi)
     idx = np.random.choice(len(midi), int(test_ratio * len(midi)), replace=False)
     #print(len(idx))
     for i in idx:
          shutil.move(os.path.join(ROOT_PATH, 'MIDI/{}/{}_train/origin_midi/{}'.format(genru,genru,midi[i])),
                    os.path.join(ROOT_PATH, 'MIDI/{}/{}_test/origin_midi/{}'.format(genru,genru,midi[i])))

shuffle_data("classic")
shuffle_data("jazz")
shuffle_data("pop")