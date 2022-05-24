from __future__ import print_function
import os
import json
import errno
from pypianoroll import Multitrack, Track, StandardTrack
import pretty_midi
import shutil
import pypianoroll
from pretty_midi import Note
import numpy as np
import sys

args = sys.argv
GENRU = args[1]
MODE = args[2]

ROOT_PATH = '/Users/stakasuka/Documents/cycleGAN/'
converter_path = os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/converter'.format(GENRU,GENRU,MODE))
cleaner_path = os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/cleaner'.format(GENRU,GENRU,MODE))

RESOLUTION = 12



def make_sure_path_exists(path):
    """与えられたパスが存在しない場合、すべての中間レベルのディレクトリを作成する"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_midi_path(root):
    """rootのディレクトリ内にあるMIDIファイルのパスのリストを再起的に返します"""
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths


def get_midi_info(pm):
    """pretty_midi.PrettyMIDI インスタンスから有用な情報を返す"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        #鳴り始めの秒数を基準にソート
        first_beat_time = pm.time_signature_changes[0].time
        #リストの先頭のmidファイルの鳴り始めの秒数を取得
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()
    #テンポの変化を4分音符/secで表したものとその時間を配列で返す
    #tc_times テンポが変化した時間
    #tempi tc_timesの各時点でのテンポを4分音符/secで表したもの

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info


def midi_filter(midi_info):
    """適合するmidiファイルに対してはTrueを、不要なものに対してはFalseを返します"""
    #1拍目が無音
    if midi_info['first_beat_time'] > 0.0:
        return False
    #tempoが変化
    elif midi_info['num_time_signature_change'] > 1:
        return False
    #4/4拍子ではない
    elif midi_info['time_signature'] not in ['4/4']:
        return False
    return True


def get_merged(multitrack):
    """ピアノロールを5つのトラック（Bass、Drums、Guitar、Piano、Strings）に結合した'pypianoroll.Multitrack'インスタンスを返す"""
    category_list = {'Bass': [], 'Drums': [], 'Guitar': [], 'Piano': [], 'Strings': []}
    program_dict = {'Piano': 0, 'Drums': 0, 'Guitar': 24, 'Bass': 32, 'Strings': 48}

    for idx, track in enumerate(multitrack.tracks):
        #enumerate() listのindexと要素を取得
        if track.is_drum:
            category_list['Drums'].append(idx)
        elif track.program//8 == 0:
            category_list['Piano'].append(idx)
        elif track.program//8 == 3:
            category_list['Guitar'].append(idx)
        elif track.program//8 == 4:
            category_list['Bass'].append(idx)
        else:
            category_list['Strings'].append(idx)
    
    tracks = []

    for key in category_list:
        merged = np.zeros(((multitrack.downbeat.size),(128)),dtype = 'uint8')
        if category_list[key]:
            for i in category_list[key]:
                merged += multitrack.tracks[i].pianoroll
            tracks.append(Track(name = '', program = program_dict[key], is_drum = (key == 'Drums'), pianoroll = merged))
        
        else:
            tracks.append(Track(name = '', program = program_dict[key], is_drum = (key == 'Drums'), pianoroll = None))

    return Multitrack(multitrack.name, multitrack.resolution, multitrack.tempo, multitrack.downbeat, tracks)

def converter(filepath):
    """MIDIファイルから変換されたマルチトラックのピアノロールをターゲットの データセットディレクトリに保存し、MIDI情報を`midi_dict`に更新する"""
    try:
        midi_name = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(resolution=RESOLUTION, name=midi_name)
        pm = pretty_midi.PrettyMIDI(filepath,resolution=RESOLUTION)
        midi_info = get_midi_info(pm)
        multitrack = pypianoroll.from_pretty_midi(pm,resolution=RESOLUTION)
        merged = get_merged(multitrack)

        make_sure_path_exists(converter_path)
        merged.save(os.path.join(converter_path, midi_name + '.npz'))

        return [midi_name, midi_info]

    except:
       return None


def main():
    """コンバーターの主な機能"""
    midi_paths = get_midi_path(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/origin_midi'.format(GENRU,GENRU,MODE)))
    midi_dict = {}
    kv_pairs = [converter(midi_path) for midi_path in midi_paths]

    for kv_pair in kv_pairs:
        if kv_pair is not None:
            midi_dict[kv_pair[0]] = kv_pair[1]

    with open(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/midis.json'.format(GENRU,GENRU,MODE)), 'w') as outfile:
        json.dump(midi_dict, outfile)

    print("[Done] {} files out of {} have been successfully converted".format(len(midi_dict), len(midi_paths)))

    with open(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/midis.json'.format(GENRU,GENRU,MODE))) as infile:
        midi_dict = json.load(infile)
    count = 0
    make_sure_path_exists(cleaner_path)
    midi_dict_clean = {}
    for key in midi_dict:
        if midi_filter(midi_dict[key]):
            midi_dict_clean[key] = midi_dict[key]
            count += 1
            shutil.copyfile(os.path.join(converter_path, key + '.npz'),
                            os.path.join(cleaner_path, key + '.npz'))

    with open(os.path.join(ROOT_PATH, 'MIDI/{}/{}_{}/midis_clean.json'.format(GENRU,GENRU,MODE)), 'w') as outfile:
        json.dump(midi_dict_clean, outfile)

    print("[Done] {} files out of {} have been successfully cleaned".format(count, len(midi_dict)))

if __name__ == "__main__":
    main()
