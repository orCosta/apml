import pickle
import numpy as np
from scipy.misc import logsumexp

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = 'RARE'

def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.
    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])
    print(pos)


class Baseline(object):
    '''
    The baseline model.
    '''
    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.i2pos = dict((v, k) for k, v in self.pos2i.items())
        # The emission probabilities:
        self.emission_prob = np.zeros((self.pos_size, self.words_size + 1)) # +1 for the "rare" words
        # The  multinomial probabilities:
        self.pos_prob = np.zeros(self.pos_size)
        self.word2i.update({RARE_WORD: self.words_size})
        # activate mle
        baseline_mle(training_set, self)

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        tag_sequences = []
        for sentence in sentences:
            esti_pos = []
            for i in range(len(sentence)):
                pos_ind = self.emission_prob[:, self.word2i[sentence[i]]] * self.pos_prob
                max_pos = np.argmax(pos_ind, axis=0)
                p = self.i2pos[max_pos]
                esti_pos.append(p)
            tag_sequences.append(esti_pos)
        return tag_sequences

		
def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.
    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    for pos, s in training_set:
        for i in range(len(s)):
            model.emission_prob[model.pos2i[pos[i]], model.word2i[s[i]]] += 1

    # Normalization
    model.pos_prob = np.sum(model.emission_prob, axis=1)
    model.pos_prob = np.nan_to_num(model.pos_prob / np.sum(model.pos_prob))

    emission_sum = np.sum(model.emission_prob, axis=0)
    model.emission_prob = np.divide(model.emission_prob, emission_sum , out=np.zeros(model.emission_prob.shape), where=(emission_sum  != 0))


class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''
    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.word2i.update({'RARE': self.words_size})
        self.pos2i.update({'Start': self.pos_size})
        self.pos2i.update({'End': self.pos_size})
        self.i2pos = dict((v, k) for k, v in self.pos2i.items())
        self.i2word = dict((v, k) for k, v in self.word2i.items())
        self.emission_prob = np.zeros((self.pos_size, self.words_size + 1)) # +1 for the "rare" words
        self.transition_prob = np.zeros((self.pos_size + 1, self.pos_size + 1)) # +1 for Start and End in each dimension
        # activate the MLE
        hmm_mle(training_set, self)

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''
        sequences = []
        for i in range(n):
            current_pos_idx = self.pos2i['Start']
            seq = []
            while(True):
                next_pos_prob = self.transition_prob[current_pos_idx]
                # Random selection of the next pos, the selection based on the transmission probs.
                next_pos_idx = np.random.choice(len(next_pos_prob), 1, p=next_pos_prob)[0]
                if(next_pos_idx == self.pos2i['End']):
                    break
                # choose one of the words that can by fit to the current tag.
                next_words_options = np.argwhere(self.emission_prob[next_pos_idx] != 0).flatten()
                next_word_idx = np.random.choice(next_words_options, 1)[0]
                seq.append(self.i2word[next_word_idx])
                current_pos_idx = next_pos_idx
            sequences.append(seq)
        return sequences

    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        epsilon = 0.000001 #this value is added to the probs in order to avoid zero division after applying log.
        tag_sequences = []
        n = self.pos_size
        trans_prob = self.transition_prob + epsilon
        emis_prob = self.emission_prob +epsilon
        for k, sentence in enumerate(sentences):
            # print("next sentence...")                              #DEBUG MSG
            t = len(sentence)
            DP_matrix = np.zeros((t, n, 2))
            esti_pos = []
            #init the first values (the transmission probs from 'start' to the next pos):
            DP_matrix[0, :, 0] = np.log(trans_prob[self.pos2i['Start'], :-1]) + np.log(emis_prob[:, self.word2i[sentence[0]]])
            for i in range(1, t):
                next_word_idx = self.word2i[sentence[i]]
                if(np.sum(self.emission_prob[:, next_word_idx]) == 0):
                    next_word_idx = self.word2i['RARE']
                    # print("find rare", k)                         #DEBUG MSG
                for j in range(n):
                    temp = DP_matrix[i-1, :, 0] + np.log(trans_prob[:-1, j]) + np.log(emis_prob[j, next_word_idx])
                    #save the max value
                    DP_matrix[i, j, 0] = np.max(temp)
                    #save the index of the previous pos
                    DP_matrix[i, j, 1] = np.argmax(temp)
            # Retrieve the pos from the last one to the first one.
            pos_idx = np.argmax(DP_matrix[-1, :, 0])
            for i in range(t-1, -1, -1): #(t-1) the last pos to 0 the first pos.
                esti_pos.append(self.i2pos[pos_idx])
                next_pos_idx = DP_matrix[i, pos_idx, 1]
                pos_idx = int(next_pos_idx)

            esti_pos.reverse()
            tag_sequences.append(esti_pos)
        return tag_sequences


def hmm_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.
    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    for pos, s in training_set:
        for i in range(len(s)):
            model.emission_prob[model.pos2i[pos[i]], model.word2i[s[i]]] += 1
            if i < len(s)-1:
                model.transition_prob[model.pos2i[pos[i]], model.pos2i[pos[i+1]]] += 1
        # add START tag
        model.transition_prob[model.pos2i['Start'], model.pos2i[pos[0]]] += 1
        # add END tag
        model.transition_prob[model.pos2i[pos[-1]], model.pos2i['End']] += 1
    # Normalize the probs
    emission_sum = np.sum(model.emission_prob, axis=0)
    model.emission_prob = np.divide(model.emission_prob, emission_sum , out=np.zeros(model.emission_prob.shape), where=(emission_sum  != 0))
    model.transition_prob = (model.transition_prob.T / np.sum(model.transition_prob, axis=1)).T
    model.transition_prob = np.nan_to_num(model.transition_prob)


class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''
    def __init__(self, pos_tags, words, training_set, phi):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.phi = phi
        self.pos2i.update({'Start': self.pos_size})
        self.word2i.update({'RARE': self.words_size})
        self.i2pos = dict((v, k) for k, v in self.pos2i.items())

    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''
        tag_sequences = []
        n = self.pos_size

        for sentence in sentences:
            t = len(sentence)
            DP_matrix = np.zeros((t, n, 2)) #The second dim is for keeping the index in each step of the DP.
            esti_pos = []
            # init first values:
            for j in range(n):
                DP_matrix[0, j, 0] = np.sum(w[self.phi(sentence[0], self.i2pos[j], 'Start', self)])
            # DP_matrix[0, :, 0] -= logsumexp(DP_matrix[0, :, 0])

            for i in range(1, t):
                for j in range(n):
                    temp = np.zeros(n)
                    for k in range(n):
                        temp[k] = np.sum(w[self.phi(sentence[i], self.i2pos[j], self.i2pos[k], self)])

                    # temp -= logsumexp(temp)
                    temp += DP_matrix[i - 1, :, 0]
                    DP_matrix[i, j, 0] = np.max(temp)
                    DP_matrix[i, j, 1] = np.argmax(temp)

            pos_idx = np.argmax(DP_matrix[-1, :, 0])
            for i in range(t - 1, -1, -1):  # (t-1) the last pos, to 0 the first pos.
                esti_pos.append(self.i2pos[pos_idx])
                next_pos_idx = DP_matrix[i, pos_idx, 1]
                pos_idx = int(next_pos_idx)

            esti_pos.reverse()
            tag_sequences.append(esti_pos)

        return tag_sequences


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    for e in range(epochs):
        # k = 0                                                 #DEBUG
        for pos, sentence in training_set:
            # k += 1                                            #DEBUG
            # print("next sentence..", k)                       #DEBUG
            esti_pos = initial_model.viterbi([sentence], w0)[0]
            w0[initial_model.phi(sentence[0], pos[0], 'Start', initial_model)] += eta
            w0[initial_model.phi(sentence[0], esti_pos[0], 'Start', initial_model)] -= eta
            for j in range(1, len(sentence)):
                w0[initial_model.phi(sentence[j], pos[j], pos[j-1], initial_model)] += eta
                w0[initial_model.phi(sentence[j], esti_pos[j], esti_pos[j-1], initial_model)] -= eta
    return w0


def phi_1(x, y, y_prev, model):
    '''
    This feature function base on the basic probabilities of the transmission and emission.
    :param X: a word
    :param y: the pos tag for the word.
    :param y_prev: the tag for the previous word.
    :param model: the MEMM model.
    :return: a list of indices that have a "1" in the binary feature vector.
    '''
    indices = []
    indices.append((model.pos2i[y_prev] * model.pos_size) + model.pos2i[y])
    indices.append((model.word2i[x] * model.pos_size) + model.pos2i[y])
    return indices


def phi_2(x, y, y_prev, model):
    '''
    This feature function base on the basic probabilities of the transmission and emission.
    :param X: a word
    :param y: the pos tag for the word.
    :param y_prev: the tag for the previous word.
    :param model: the MEMM model.
    :return: a list of indices that have a "1" in the binary feature vector.
    '''
    n = model.pos_size
    d = model.words_size
    end_of_trans_idx = (n+1)*n
    end_of_emiss_idx = end_of_trans_idx + n*d
    indices = []
                                                                   # transmission prob:
    indices.append((model.pos2i[y_prev] * model.pos_size) + model.pos2i[y])
                                                                   # emission prob:
    indices.append(end_of_trans_idx + (model.word2i[x] * model.pos_size) + model.pos2i[y])
                                                                   # suffixes:
    if(x.endswith('ion') or x.endswith('age') or x.endswith('ity')):   # for N
        indices.append(end_of_emiss_idx)
    if(x.endswith('ly') or x.endswith('ful') or x.endswith('able')):   # for Adj
        indices.append(end_of_emiss_idx + 1)
    if x.endswith('ate'):                                              # for V
        indices.append(end_of_emiss_idx + 2)
                                                                    # word length:
    if (len(x) < 3):
        indices.append(end_of_emiss_idx + 3)
                                                                    # capital letters:
    if x[0].isupper():
        indices.append(end_of_emiss_idx + 4)

    if x[0].isdigit():                                              # digits
        indices.append(end_of_emiss_idx + 5)
    return indices


def compare_pos(seq1, seq2):
    '''
    compare between every tag in each sequence. return the result in percentage.
    :param seq1: sequence of pos sequences.
    :param seq2: sequence of pos sequences.
    '''
    total_words = 0
    correct_pos = 0
    for i, j in zip(seq1, seq2):
        for p1, p2 in zip(i, j):
            total_words += 1
            if p1 == p2:
                correct_pos += 1
    return correct_pos/total_words


def processing_the_data(data, words):
    '''
    preproccessing the data, convert rare words to "RARE" value in the sentences.
    :param data: pos tags and sentences.
    :return: the data
    '''
    word_hist = {word: 0 for word in words}
    pos_tags, sentences = [list(unzippedTuple) for unzippedTuple in zip(*data)]
    for sentence in sentences:
        for w in sentence:
            word_hist[w] += 1

    for sentence in sentences:
        for i, w in enumerate(sentence):
            if word_hist[w] < 4:
                sentence[i] = 'RARE'
    return data


def split_the_data(data, x):
    '''
    Split the data to test set and training set.
    :param data: pos tags and sentences.
    :param x: the size (in percentage) of the training set.
    :return: training set, test set
    '''
    size = int(len(data) * 0.9)
    training_set = data[:size]
    test_set = data[size_1:]
    return training_set, test_set


if __name__ == '__main__':

    # Load the data :
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    data = processing_the_data(data, words)
    # *********************************************************
    # ****************** MODEL COMPARISON *********************
    # *********************************************************
    size_1 = int(len(data)*0.9)
    size_2 = len(data)-int(len(data)*0.1)
    training_set = data[:size_1]
    test_set = data[size_1:]
    test_pos, test_sentences = zip(*test_set)

    # *********** MEMM model**********************
    memm_mod = MEMM(pos, words, test_set, phi_2)
    n = len(pos)
    d= len(words)
    # w = np.zeros((n+1)*n + (d+1)*n +6)                   #for Phi_2
    w = np.zeros((n + 1) * n + (d + 1) * n)                #for Phi_1

    w1 = perceptron(training_set, memm_mod, w)
    memm_pos = memm_mod.viterbi(test_sentences, w1)

    result = compare_pos(test_pos, memm_pos)
    print("MEMM accuracy      : ", result)

    # *********** Baseline model*******************
    b_mod = Baseline(pos, words, training_set)
    b_pos = b_mod.MAP(test_sentences)
    b_acc = compare_pos(test_pos, b_pos)
    print("baseline accuracy : ", b_acc)

    # *********** HMM model************************
    h_mod = HMM(pos, words, training_set)
    h_pos = h_mod.viterbi(test_sentences)
    h_acc = compare_pos(test_pos, h_pos)
    print("HMM accuracy      : ", h_acc)

    print("test1")