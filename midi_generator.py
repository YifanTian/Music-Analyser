#!/bin/python

from __future__ import print_function

from music_model import MusicModel
import random
from math import log
import numpy as np
import data
import argparse

from music_model import Unigram
from music_model import Smoothing_Bigram
from music_model import Kneser_Ney_Bigram
from music_model import Smoothing_Trigram
from music_model import Back_off_Trigram
from music_model import Interpolation_Trigram

from util import compose
from util import play_midi
from util import plot_midi

class Sampler:

    def __init__(self, lm, temp = 1.0):
        """Sampler for a given language model.

        Supports the use of temperature, i.e. how peaky we want to treat the
        distribution as. Temperature of 1 means no change, temperature <1 means
        less randomness (samples high probability words even more), and temp>1
        means more randomness (samples low prob words more than otherwise). See
        simulated annealing for what this means.
        """
        self.lm = lm
        self.rnd = random.Random()
        self.temp = temp

    def sample_sentence(self, prefix = [], max_length = 100):
        """Sample a random sentence (list of words) from the language model.

        Samples words till either EOS symbol is sampled or max_length is reached.
        Does not make any assumptions about the length of the context.
        """
        i = 0
        sent = prefix
        word = self.sample_next(sent, False)
        # print("word: ",word)
        while i <= max_length and word != "END_OF_SENTENCE":
            sent.append(word)
            word = self.sample_next(sent)
            i += 1
        return sent

    def sample_next(self, prev, incl_eos = True):
        """Samples a single word from context.

        Can be useful to debug the model, for example if you have a bigram model,
        and know the probability of X-Y should be really high, you can run
        sample_next([Y]) to see how often X get generated.

        incl_eos determines whether the space of words should include EOS or not.
        """
        wps = []
        tot = -np.inf # this is the log (total mass)
        for w in self.lm.vocab():
            if not incl_eos and w == "END_OF_SENTENCE":
                continue
            lp = self.lm.cond_logprob(w, prev)                      # log sum
            wps.append([w, lp/self.temp])
            tot = np.logaddexp2(lp/self.temp, tot)
        p = self.rnd.random()
        word = self.rnd.choice(wps)[0]
        s = -np.inf # running mass
        for w,lp in wps:
            s = np.logaddexp2(s, lp)
            if p < pow(2, s-tot):
                word = w
                break
        return word

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some midi.')
    parser.add_argument('midi_name', type=str, help='midi name')

    # model = Unigram()
    model = Smoothing_Bigram()
    # model = Kneser_Ney_Bigram()
    # model = Smoothing_Trigram()
    # model = Back_off_Trigram()
    # model = Interpolation_Trigram()
    model_name = model.model_name
    print(model_name)

    args = parser.parse_args()
    midi_file = args.midi_name

    melody_root = midi_file.split('/')[-1][:-4]
    corpus = [data.read_pitchs(midi_file)]
    model.fit_corpus(corpus)
    sampler = Sampler(model)
    save_dir = './'
    
    for i in range(1):
        song_pitchs = sampler.sample_sentence([])
        song_name = "{}{}_{}_{}.mid".format(save_dir,melody_root,model_name,i)
        naive_song = compose(song_pitchs, song_name, save_dir)
        play_midi(naive_song)
