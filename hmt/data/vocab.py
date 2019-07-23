class Vocab:
    def __init__(self, add_pad=False):
        self.itow = {}
        self.wtoi = {}
        self.num = 0
        if add_pad:
            self.add_word('<pad>')

    def __len__(self):
        return len(self.wtoi)

    def get_word(self, indices):
        return self.itow[indices]

    def get_word_id(self, word):
        return self.wtoi(word)

    def add_word(self, word):
        if word not in self.wtoi:
            self.wtoi[word] = self.num
            self.itow[self.num] = word
            self.num += 1

    def str_to_indices(self, sent):
        if isinstance(sent, str):
            return self.wtoi[sent]
        else:
            return [self.wtoi[s] for s in sent]

    def pad(self):
        return self.wtoi['<pad>']
