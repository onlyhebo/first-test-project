class Vocab:
    def __init__(self):
        self.itow = {}
        self.wtoi = {}
        self.num = 0

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
        return [self.wtoi[s] for s in sent]

