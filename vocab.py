class Vocab:
    def __init__(self, vocab=[]):
        self.vocab_size = 0
        self.nodes = {}  # dictionary
        for node in vocab:
            self.nodes[node] = self.vocab_size
            self.vocab_size += 1

    def nodes2ids(self, nodes_list: list):
        ids = []
        ids = [self.nodes[i] for i in nodes_list]
        return ids