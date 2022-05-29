"""
Some codes from https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer
"""

from random import shuffle
import numpy as np
import os
import shutil
import pretty_midi
from pypianoroll import Multitrack, Track
import pypianoroll
from utils import *
import convert_clean
import tonnetz
import sys

args = sys.argv
GENRU = args[1]
MODE = args[2]

#your directory
ROOT_PATH = '/'

RESOLUTION = 12


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % (RESOLUTION*16)) != 0:
        piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % (RESOLUTION*16)):], axis=0)
    piano_roll = piano_roll.reshape(-1, RESOLUTION*16, 128)
    return piano_roll

def to_binary(bars, threshold=0.0):
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


"""1. convert_clean.py"""
convert_clean.main()


"""2. choose the clean midi from original sets"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi'.format(GENRU,GENRU,MODE))):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi'.format(GENRU,GENRU,MODE)))
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner').format(GENRU,GENRU,MODE))]
for i in l:
    shutil.copy(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/origin_midi'.format(GENRU,GENRU,MODE), os.path.splitext(i)[0] + '.mid'),
                os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi'.format(GENRU,GENRU,MODE), os.path.splitext(i)[0] + '.mid'))


"""3. merge and crop"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi_gen'.format(GENRU,GENRU,MODE))):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi_gen'.format(GENRU,GENRU,MODE)))
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_npy'.format(GENRU,GENRU,MODE))):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_npy'.format(GENRU,GENRU,MODE)))
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi'.format(GENRU,GENRU,MODE)))]
count = 0
np.set_printoptions(threshold=np.inf)
for i in range(len(l)):
    try:
        multitrack = Multitrack(resolution=RESOLUTION, name=os.path.splitext(l[i])[0])
        x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi'.format(GENRU,GENRU,MODE), l[i]), resolution = RESOLUTION)
        multitrack = pypianoroll.from_pretty_midi(x,resolution = RESOLUTION)

        category_list = {'Piano': [], 'Drums': []}
        program_dict = {'Piano': 0, 'Drums': 0}

        for idx, track in enumerate(multitrack.tracks):
            if track.is_drum:
                category_list['Drums'].append(idx)
            else:
                category_list['Piano'].append(idx)
        tracks = []

        merged = np.zeros(((multitrack.downbeat.size),(128)),dtype = 'uint8')

        for index in category_list['Piano']:
            merged += multitrack.tracks[index].pianoroll
        
        pr = get_bar_piano_roll(merged)
        
        pr_clip = pr[:, :, 24:108]
        if int(pr_clip.shape[0] % 4) != 0:
            pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
        pr_re = pr_clip.reshape(-1, RESOLUTION*16, 84, 1)
        pr_rere = np.split(pr_re,(pr_re.shape[0]))
        save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_midi_gen'.format(GENRU,GENRU,MODE), os.path.splitext(l[i])[0] + '.mid'),100)
        np.save(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner_npy'.format(GENRU,GENRU,MODE), os.path.splitext(l[i])[0] + '.npy'), pr_re)
    except:
        count += 1
        print('Wrong', l[i])
        continue


"""4. chord detection"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE))):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE)))
tonnetz.detect_chord_key()


"""5. concatenate into a big binary numpy array file"""
l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE)))]
train = np.load(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE), l[0]))
for i in range(1, len(l)):
    t = np.load(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/chord_npy'.format(GENRU,GENRU,MODE), l[i]))
    train = np.concatenate((train, t), axis=0)
np.save(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/{}_{}_piano.npy'.format(GENRU,GENRU,MODE,GENRU,MODE)), (train > 0.0))

"""6. separate numpy files into a single phrase"""

if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/phrase_test'.format(GENRU,GENRU,MODE))):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/phrase_test'.format(GENRU,GENRU,MODE)))

x = np.load(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/{}_{}_piano.npy'.format(GENRU,GENRU,MODE,GENRU,MODE)))
count = 0
print(x.shape)
for i in range(x.shape[0]):
    if np.max(x[i]):
        count += 1
        np.save(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/phrase_test/{}_piano_{}_{}.npy'.format(GENRU,GENRU,MODE,GENRU,MODE,i+1)), x[i])
print(count)