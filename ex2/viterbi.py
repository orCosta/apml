     def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        tag_sequences = []
        n = self.pos_size
        trans_prob = self.transition_prob + 1
        emis_prob = self.emission_prob + 1
        for sentence in sentences:
            t = len(sentence)
            DP_matrix = np.zeros((t, n, 2))
            esti_pos = []
            #init the first values (the transmission probs from 'start' to the next pos):
            DP_matrix[0, :, 0] = np.log(trans_prob[-1]) + np.log(emis_prob[:, self.word2i[sentence[0]]])
            for i in range(1, t):
                for j in range(n):
                    temp = DP_matrix[i-1, :, 0] + np.log(trans_prob[:-1, j]) + np.log(emis_prob[j, self.word2i[sentence[i]]])
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