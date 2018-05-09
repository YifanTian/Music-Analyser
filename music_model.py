#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
import copy

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class MusicModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        print("finish fitting midi files")
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        test = False
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s,test)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, test):
        p = 0.0
        for i in xrange(len(sentence)):
            if test:
                print(sentence[i], sentence[:i], self.cond_logprob(sentence[i], sentence[:i]))
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)
        self.method_name = "Unigram"
        self.missing_word = 0
        self.model_name = 'Unigram'
        
    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            # print("words missing!")
            self.missing_word += 1
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

    def print_name(self):
        print("Unigram")

    def reset_missing(self):
        self.missing_word = 0

    def print_missing(self):
        print("missing_word:", self.missing_word)


class Smoothing_Bigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.model_tot = dict()
        self.vocab_dict = dict()
        self.lbackoff = log(backoff, 2)
        self.method_name = "Smoothing_Bigram"
        self.missing_word = 0
        self.missing_prefix = 0
        
        self.smooth_factor = 0.0         # best: between 0.01 to 0.001
        self.model_name = 'Smoothing_Bigram'
        self.bigram = dict()

    def inc_bigram(self, w):
        if w[0] in self.model:
            if w[1] in self.model[w[0]]:
                self.model[w[0]][w[1]] += 1.0
            else:
                self.model[w[0]][w[1]] = 1.0
        else:
            self.model[w[0]] = dict()
            self.model[w[0]][w[1]] = 1

    def fit_sentence(self, sentence):
        # self.inc_bigram(('START',sentence[0]))
        self.inc_bigram((0,sentence[0]))
        for i in range(len(sentence)-1):                    # pair of words
            self.inc_bigram((sentence[i],sentence[i+1]))
            # print(sentence[i],sentence[i+1])
        # self.inc_bigram((sentence[-1],'END_OF_SENTENCE'))
        self.inc_bigram((sentence[-1],100))
        for w in sentence:
            self.inc_word(w)
        # self.inc_word('END_OF_SENTENCE')
        self.inc_word(100)
        # self.bigram = self.model.copy()
        self.bigram = copy.deepcopy(self.model)
        # raise SystemExit(0)

    def norm(self):
        """Normalize and convert to log2-probs."""
        k = self.smooth_factor
        for prefix in self.model:
            tot = 0.0
            for word in self.model[prefix]:
                tot += self.model[prefix][word]
            tot += k*len(self.vocab_dict)
            ltot = log(tot, 2)
            self.model_tot[prefix] = ltot
            for word in self.model[prefix]:
                # if log(self.model[prefix][word]) - ltot < -14:
                if prefix == "original":
                    print(prefix, word, log(self.model[prefix][word]) - ltot, log(self.model[prefix][word] + k*1, 2) - ltot, log(k*1, 2) - ltot)
                self.model[prefix][word] = log(self.model[prefix][word] + k*1, 2) - ltot

    def cond_logprob(self, word, previous):
        """ """
        if previous == []:
            previous = ['START']
        if previous[-1] in self.model:
            if word in self.model[previous[-1]]:
                return self.model[previous[-1]][word]
            elif word in self.vocab_dict and self.smooth_factor > 0:
                return log(self.smooth_factor*1, 2)-self.model_tot[previous[-1]]
            else:
                self.missing_word += 1
                return self.lbackoff
        else:
            self.missing_prefix += 1
            return self.lbackoff

    def inc_word(self, w):
        if w in self.vocab_dict:
            self.vocab_dict[w] += 1.0
        else:
            self.vocab_dict[w] = 1.0

    def vocab(self):
        return self.vocab_dict.keys()

    def print_name(self):
        print("Smoothing_Bigram")

    def reset_missing(self):
        self.missing_word = 0
        self.missing_prefix = 0

    def print_missing(self):
        print("missing_word:", self.missing_word)
        print("missing_prefix:", self.missing_prefix)


class Kneser_Ney_Bigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.model_tot = dict()
        self.vocab_dict = dict()
        self.lbackoff = log(backoff, 2)
        self.Pcontinuation = dict()
        self.Pcontinuation_val = dict()
        self.total_uniques = 0
        self.d = 0.75
        self.method_name = "Kneser_Ney_Bigram"
        self.missing_word = 0
        self.missing_prefix = 0
        
        self.smooth_factor = 0.001         # best: between 0.01 to 0.001
        self.model_name = 'Kneser_Ney_Bigram'

    def inc_bigram(self, w):
        """ """
        if w[1] in self.Pcontinuation:
            if w[0] not in self.Pcontinuation[w[1]]:
                self.Pcontinuation[w[1]][w[0]] = 1.0
        else:
            self.Pcontinuation[w[1]] = dict()
            self.Pcontinuation[w[1]][w[0]] = 1.0

        if w[0] in self.model:
            if w[1] in self.model[w[0]]:
                self.model[w[0]][w[1]] += 1.0
            else:
                self.model[w[0]][w[1]] = 1.0
        else:
            self.model[w[0]] = dict()
            self.model[w[0]][w[1]] = 1

    def fit_sentence(self, sentence):
        """ """
        # self.inc_bigram((sentence[i],sentence[i+1]))
        self.inc_bigram(('START',sentence[0]))
        for i in range(len(sentence)-1):                    # pair of words
            self.inc_bigram((sentence[i],sentence[i+1]))
        self.inc_bigram((sentence[-1],'END_OF_SENTENCE'))
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        for w in self.Pcontinuation:
            self.total_uniques += len(self.Pcontinuation[w]) 
        for w in self.vocab_dict:
            if w in self.Pcontinuation:
                self.Pcontinuation_val[w] = len(self.Pcontinuation[w])/self.total_uniques
            else:
                self.Pcontinuation[w] = 0;
                self.Pcontinuation_val[w] = 0;

        k = self.smooth_factor
        for prefix in self.model:
            tot = 0.0
            for word in self.model[prefix]:
                tot += self.model[prefix][word]
            self.model_tot[prefix] = tot
            for word in self.model[prefix]:
                lamda = (self.d/tot)*len(self.model[prefix])
                self.model[prefix][word] = max(self.model[prefix][word] - self.d,0)/tot + lamda*self.Pcontinuation_val[word]
                self.model[prefix][word] = log(self.model[prefix][word],2)


    def cond_logprob(self, word, previous):
        """ """
        if previous == []:
            previous = ['START']
        if previous[-1] in self.model:
            if word in self.model[previous[-1]]:
                return self.model[previous[-1]][word]
            elif word in self.vocab_dict and self.Pcontinuation_val[word] > 0:    
                return log((self.d/self.model_tot[previous[-1]])*len(self.model[previous[-1]])*self.Pcontinuation_val[word], 2)
            else:
                return self.lbackoff
                self.missing_word += 1
        else:
            return self.lbackoff
            self.missing_prefix += 1

    def inc_word(self, w):
        if w in self.vocab_dict:
            self.vocab_dict[w] += 1.0
        else:
            self.vocab_dict[w] = 1.0

    def vocab(self):
        return self.vocab_dict.keys()

    def print_name(self):
        print("Kneser_Ney_Bigram")

    def log(self):
        # open("Kneser_Ney_Bigram_lm_log.txt",w)
        return


class Smoothing_Trigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.vocab_dict = dict()
        self.lbackoff = log(backoff, 2)
        self.model_tot = dict()
        self.method_name = "Smoothing_Trigram"
        self.missing_word = 0
        self.missing_prefix = 0

        self.smooth_factor = 0         # best: between 0.01 to 0.001
        self.model_name = 'Smoothing_Trigram'

    def inc_trigram(self, pre, w):
        if pre in self.model:
            if w in self.model[pre]:
                self.model[pre][w] += 1.0
            else:
                self.model[pre][w] = 1.0
        else:
            self.model[pre] = dict()
            self.model[pre][w] = 1

    def fit_sentence(self, sentence):
        self.inc_trigram(('START','START'),sentence[0])
        if len(sentence) > 1:
            self.inc_trigram(('START',sentence[0]),sentence[1])
            for i in range(len(sentence)-2):                    # pair of words
                self.inc_trigram((sentence[i],sentence[i+1]),sentence[i+2])
            self.inc_trigram((sentence[-2],sentence[-1]),'END_OF_SENTENCE')
            for w in sentence:
                self.inc_word(w)
            self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        k = self.smooth_factor
        for prefix in self.model:
            tot = 0.0
            for word in self.model[prefix]:
                tot += self.model[prefix][word]
            tot += k*len(self.vocab_dict)
            ltot = log(tot, 2)
            self.model_tot[prefix] = ltot
            for word in self.model[prefix]:
                self.model[prefix][word] = log(self.model[prefix][word] + k*1, 2) - ltot

    def cond_logprob(self, word, previous):
        """ """
        if previous == []:
            previous = ['START','START']
        elif len(previous) == 1:
            previous = ['START',previous[-1]]
        pre = tuple(previous[-2:])
        if pre in self.model:
            if word in self.model[pre]:
                return self.model[pre][word]
            elif word in self.vocab_dict and self.smooth_factor > 0:
                return log(self.smooth_factor*1, 2)-self.model_tot[pre]
            else:
                self.missing_word += 1
                return self.lbackoff
        else:
            self.missing_prefix += 1
            return self.lbackoff

    def inc_word(self, w):
        if w in self.vocab_dict:
            self.vocab_dict[w] += 1.0
        else:
            self.vocab_dict[w] = 1.0

    def vocab(self):
        return self.vocab_dict.keys()

    def print_name(self):
        print("Smoothing_Trigram")

    def reset_missing(self):
        self.missing_word = 0
        self.missing_prefix = 0

    def print_missing(self):
        print("missing_word:", self.missing_word)
        print("missing_prefix:", self.missing_prefix)


class Back_off_Trigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.trigram_model = dict()
        self.bigram_model = dict()
        self.unigram_model = dict()
        self.vocab_dict = dict()
        self.lbackoff = log(backoff, 2)
        self.model_tot = dict()
        self.method_name = "Back_off_Trigram"

        # self.smooth_factor = 0.0001         # best: between 0.01 to 0.001
        self.model_name = 'Back_off_Trigram'

    def inc_unigram(self, w):
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def inc_bigram(self, w):
        if w[0] in self.bigram_model:
            if w[1] in self.bigram_model[w[0]]:
                self.bigram_model[w[0]][w[1]] += 1.0
            else:
                self.bigram_model[w[0]][w[1]] = 1.0
        else:
            self.bigram_model[w[0]] = dict()
            self.bigram_model[w[0]][w[1]] = 1

    def inc_trigram(self, pre, w):
        if pre in self.trigram_model:
            if w in self.trigram_model[pre]:
                self.trigram_model[pre][w] += 1.0
            else:
                self.trigram_model[pre][w] = 1.0
        else:
            self.trigram_model[pre] = dict()
            self.trigram_model[pre][w] = 1

    def fit_sentence(self, sentence):
        self.inc_trigram(('START','START'),sentence[0])
        if len(sentence) > 1:
            self.inc_trigram(('START',sentence[0]),sentence[1])
            for i in range(len(sentence)-2):                    # pair of words
                self.inc_trigram((sentence[i],sentence[i+1]),sentence[i+2])
            self.inc_trigram((sentence[-2],sentence[-1]),'END_OF_SENTENCE')
        
        self.inc_bigram(('START',sentence[0]))
        for i in range(len(sentence)-1):                    # pair of words
            self.inc_bigram((sentence[i],sentence[i+1]))
        self.inc_bigram((sentence[-1],'END_OF_SENTENCE'))

        self.inc_unigram('START')
        for w in sentence:
            self.inc_unigram(w)
        self.inc_unigram('END_OF_SENTENCE')

        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""

        tot = 0.0
        for word in self.unigram_model:
            tot += self.unigram_model[word]
        ltot = log(tot, 2)
        for word in self.unigram_model:
            self.unigram_model[word] = log(self.unigram_model[word],2) - ltot

        for prefix in self.bigram_model:
            tot = 0.0
            for word in self.bigram_model[prefix]:
                tot += self.bigram_model[prefix][word]
            ltot = log(tot, 2)
            # self.model_tot[prefix] = ltot
            for word in self.bigram_model[prefix]:
                self.bigram_model[prefix][word] = log(self.bigram_model[prefix][word], 2) - ltot
                # self.bigram_model[prefix][word] = self.bigram_model[prefix][word]/tot

        for prefix in self.trigram_model:
            tot = 0.0
            for word in self.trigram_model[prefix]:
                tot += self.trigram_model[prefix][word]
            ltot = log(tot, 2)
            for word in self.trigram_model[prefix]:
                self.trigram_model[prefix][word] = log(self.trigram_model[prefix][word], 2) - ltot
                # self.trigram_model[prefix][word] = self.trigram_model[prefix][word]/tot

    def cond_logprob(self, word, previous):
        """ """
        # if previous == []:
        #     if word in self.unigram_model:
        #         return self.unigram_model[word]
        #     else:
        #         return self.lbackoff
        if previous == []:
            previous = ['START','START']
        elif len(previous) == 1:
            previous = ['START',previous[-1]]
        pre = tuple(previous[-2:])
        answer = 0.0
        if pre in self.trigram_model and word in self.trigram_model[pre]:
            # if word in self.trigram_model[pre]:
            answer = self.trigram_model[pre][word]
        elif previous[-1] in self.bigram_model and word in self.bigram_model[previous[-1]]:
            answer = self.bigram_model[previous[-1]][word]
        elif word in self.unigram_model:
            answer = self.unigram_model[word]
        else:    
            answer = self.lbackoff
        return answer

    def inc_word(self, w):
        if w in self.vocab_dict:
            self.vocab_dict[w] += 1.0
        else:
            self.vocab_dict[w] = 1.0

    def vocab(self):
        return self.vocab_dict.keys()

    def print_name(self):
        print("Back_off_Trigram")




class Interpolation_Trigram(MusicModel):
    def __init__(self, backoff = 0.000001):
        self.trigram_model = dict()
        self.bigram_model = dict()
        self.unigram_model = dict()
        self.vocab_dict = dict()
        self.lbackoff = log(backoff, 2)
        self.model_tot = dict()
        self.l1 = 0.1
        self.l2 = 0.3
        self.l3 = 0.6
        self.model_name = "Interpolation_Trigram"

        # self.smooth_factor = 0.0001         # best: between 0.01 to 0.001

    def inc_unigram(self, w):
        if w in self.unigram_model:
            self.unigram_model[w] += 1.0
        else:
            self.unigram_model[w] = 1.0

    def inc_bigram(self, w):
        if w[0] in self.bigram_model:
            if w[1] in self.bigram_model[w[0]]:
                self.bigram_model[w[0]][w[1]] += 1.0
            else:
                self.bigram_model[w[0]][w[1]] = 1.0
        else:
            self.bigram_model[w[0]] = dict()
            self.bigram_model[w[0]][w[1]] = 1

    def inc_trigram(self, pre, w):
        if pre in self.trigram_model:
            if w in self.trigram_model[pre]:
                self.trigram_model[pre][w] += 1.0
            else:
                self.trigram_model[pre][w] = 1.0
        else:
            self.trigram_model[pre] = dict()
            self.trigram_model[pre][w] = 1

    def fit_sentence(self, sentence):
        self.inc_trigram(('START','START'),sentence[0])
        if len(sentence) > 1:
            self.inc_trigram(('START',sentence[0]),sentence[1])
            for i in range(len(sentence)-2):                    # pair of words
                self.inc_trigram((sentence[i],sentence[i+1]),sentence[i+2])
            self.inc_trigram((sentence[-2],sentence[-1]),'END_OF_SENTENCE')
        
        self.inc_bigram(('START',sentence[0]))
        for i in range(len(sentence)-1):                    # pair of words
            self.inc_bigram((sentence[i],sentence[i+1]))
        self.inc_bigram((sentence[-1],'END_OF_SENTENCE'))

        self.inc_unigram('START')
        for w in sentence:
            self.inc_unigram(w)
        self.inc_unigram('END_OF_SENTENCE')

        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""

        tot = 0.0
        for word in self.unigram_model:
            tot += self.unigram_model[word]
        # ltot = log(tot, 2)
        for word in self.unigram_model:
            self.unigram_model[word] = self.unigram_model[word]/tot

        for prefix in self.bigram_model:
            tot = 0.0
            for word in self.bigram_model[prefix]:
                tot += self.bigram_model[prefix][word]
            # ltot = log(tot, 2)
            # self.model_tot[prefix] = ltot
            for word in self.bigram_model[prefix]:
                # self.bigram_model[prefix][word] = log(self.bigram_model[prefix][word], 2) - ltot
                self.bigram_model[prefix][word] = self.bigram_model[prefix][word]/tot

        for prefix in self.trigram_model:
            tot = 0.0
            for word in self.trigram_model[prefix]:
                tot += self.trigram_model[prefix][word]
            # ltot = log(tot, 2)
            for word in self.trigram_model[prefix]:
                # self.trigram_model[prefix][word] = log(self.trigram_model[prefix][word], 2) - ltot
                self.trigram_model[prefix][word] = self.trigram_model[prefix][word]/tot

    def cond_logprob(self, word, previous):
        """ """
        # if previous == []:
        #     if word in self.unigram_model:
        #         return self.unigram_model[word]
        #     else:
        #         return self.lbackoff
        if previous == []:
            previous = ['START','START']
        elif len(previous) == 1:
            previous = ['START',previous[-1]]
        pre = tuple(previous[-2:])
        answer = 0.0
        if pre in self.trigram_model:
            if word in self.trigram_model[pre]:
                answer =  log(self.l1*self.unigram_model[word] + self.l2*self.bigram_model[previous[-1]][word] + self.l3*self.trigram_model[pre][word],2)
            elif word in self.bigram_model[previous[-1]]:
                answer =  log(self.l1*self.unigram_model[word] + self.l2*self.bigram_model[previous[-1]][word],2)
            elif word in self.unigram_model:
                answer = log(self.l1*self.unigram_model[word],2)
            else:
                answer = self.lbackoff
        elif previous[-1] in self.bigram_model:
            if word in self.bigram_model[previous[-1]]:
                answer = log(self.l1*self.unigram_model[word] + self.l2*self.bigram_model[previous[-1]][word],2)
            elif word in self.unigram_model:
                answer = log(self.l1*self.unigram_model[word],2)
            else:
                answer = self.lbackoff
        elif word in self.unigram_model:
            answer = log(self.l1*self.unigram_model[word],2)
        else:
            answer = self.lbackoff
        return answer

    def inc_word(self, w):
        if w in self.vocab_dict:
            self.vocab_dict[w] += 1.0
        else:
            self.vocab_dict[w] = 1.0

    def vocab(self):
        return self.vocab_dict.keys()

    def print_name(self):
        print("Interpolation_Trigram")


