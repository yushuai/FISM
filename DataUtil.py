import random


class DataUtil:
    def __init__(self):
        self.ratings = list()
        self.index = 0
        self.user_rated_items = {}

    def load_data(self, filename):
        isFirst = True
        row_no = 0
        for line in open(filename):
            if isFirst:
                isFirst = False
                continue
            arr = line.split()
            user_rated_items = set()
            for index, item in enumerate(arr):
                if index % 2 == 0:
                    self.ratings.append((row_no, int(item) - 1, 1))
                    user_rated_items.add(int(item - 1))
                else:
                    item = random.randint(0, 1178)
                    if item not in user_rated_items:
                        self.ratings.append((row_no, item, 0))
            self.user_rated_items[row_no] = user_rated_items
            row_no += 1

    def get_batch(self, reset=False, batch_size=16):
        if reset:
            random.shuffle(self.ratings)
            self.index = 0
        batch = self.ratings[self.index: self.index + batch_size]
        self.index += batch_size
        x_train = []
        y_train = []
        for uid, item_id, label in self.batch:
            x_train.append((uid,item_id))
            y_train.append(label)
        return x_train, y_train




