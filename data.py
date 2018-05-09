from __future__ import print_function

import midi
import sys
from music_model import MusicModel
import random
from math import log
import numpy as np
import data

from util import compose
from util import play_midi
from util import plot_midi
from os import listdir, mkdir
import matplotlib.pyplot as plt


def read_pitchs(midi_file):
    """ """
    song = midi.read_midifile(midi_file)
    song.make_ticks_abs()
    tracks = []
    for track in song:
        notes = [note for note in track if note.name == 'Note On']
        pitch = [note.pitch for note in notes]
        tick = [note.tick for note in notes]
        tracks += [tick, pitch]

    try:
        ticks = tracks[2]
        pitchs = tracks[3]
    except:
        ticks = tracks[0]
        pitchs = tracks[1]
    return pitchs


if __name__ == "__main__":
    # midi_dir = './collection/midi/'
    # midi_dir = './collection/midi-learn/'
    # midi_dir = './collection/midi_collection1/'
    # midi_dir = './collection/zhou/'
    midi_dir = './collection/scale_chords_small/midi/'

    midi_files_list = [f for f in listdir(midi_dir) if f.endswith('.mid')]
    # print(midi_files_list)
    for midi_file in midi_files_list:
        melody_root = midi_file.split('/')[-1][:-4]
        corpus = [data.read_pitchs(midi_dir+midi_file)]
        # model.fit_corpus(corpus)
        plot_midi(midi_dir+midi_file,show=False,save=True,save_dir=midi_dir)
        # plt.savefig(midi_dir+melody_root+'.png')