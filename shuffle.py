import numpy as np
import os
import shutil
from utils import *

test_ratio = 0.1
ROOT_PATH = '/Users/stakasuka/Documents/cycleGAN'

pop = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_train/origin_midi'))]
print(pop)
idx = np.random.choice(len(pop), int(test_ratio * len(pop)), replace=False)
print(len(idx))
for i in idx:
     shutil.move(os.path.join(ROOT_PATH, 'MIDI/pop/pop_train/origin_midi', pop[i]),
                 os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi', pop[i]))

jazz = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_train/origin_midi'))]
print(jazz)
idx = np.random.choice(len(jazz), int(test_ratio * len(jazz)), replace=False)
print(len(idx))
for i in idx:
     shutil.move(os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_train/origin_midi', jazz[i]),
                 os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_test/origin_midi', jazz[i]))

cla = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/origin_midi'))]
print(cla)
idx = np.random.choice(len(cla), int(test_ratio * len(cla)), replace=False)
print(len(idx))
for i in idx:
     shutil.move(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/origin_midi', cla[i]),
                 os.path.join(ROOT_PATH, 'MIDI/classic/classic_test/origin_midi', cla[i]))