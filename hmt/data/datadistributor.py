class DataDistributor():

    def __init__(self):
        self.queue = []

    def add(self, q):
        self.queue.append(q)

    def send_example_list(self, batch):
        for i in range(len(self.queue)):
            self.queue[i].put(batch[i])
