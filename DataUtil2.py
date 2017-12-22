import random
import numpy as np


class DataUtil:
    def __init__(self, dataset, userNum, itemNum):
        self.dataset = dataset
        # self.popular_product = self.count_popular_item()
        self.userNum = userNum
        self.itemNum = itemNum
        self.num_negative = 4
        self.train, self.train_map = self.get_train_instance()
        self.index = 0
        self.sample_num = len(self.train)
        self.item_rated_by_user = dict()
        self.calc_item_rated_by_user()
        self.test = {}
        self.test_answer = []
        self.read_test('dataset/ml_test')
    def calc_item_rated_by_user(self):
        for u, i in self.train_map:
            if u not in self.item_rated_by_user.keys():
                self.item_rated_by_user[u] = set()
            self.item_rated_by_user[u].add(i)
        for i in range(self.userNum):
            if len(self.item_rated_by_user[i]) == 0:
                print('%d not in rated_set' % (i))

    def next_batch(self, size=16, reset=False):
        if reset:
            self.index = 0
            random.shuffle(self.train)
        x = []
        y = []
        for t in self.train[self.index: self.index + size]:
            u, i, r = t
            x.append((u, i))
            y.append(r)
        self.index += size
        return x, y

    # sort a dictionary by value, bigger value ranks high
    def sort_by_value(self, d):
        items = d.items()
        backitems = [[v[1], v[0]] for v in items]
        backitems.sort(reverse=True)
        return [backitems[i][1] for i in range(0, len(backitems))]

    # count how many times each item was consumed, returns a sorted list
    def count_popular_item(self):
        info = dict()
        for line in open(self.dataset):
            userid, itemid = line.split('\t')
            if int(itemid) not in info:
                info[int(itemid)] = 1
            else:
                info[int(itemid)] += 1
        return self.sort_by_value(info)

    def get_train_instance(self):
        print('loading dataset')
        triple = set()
        train = set()
        for line in open(self.dataset):
            userid, itemid = line.split('\t')
            train.add((int(userid), int(itemid)))
        for line in open(self.dataset):
            userid, itemid = line.split('\t')
            triple.add((int(userid), int(itemid), 1))
            #   append one negative instance drawing from popular set
            #   for p_product in self.popular_product:
            #     if (int(userid), int(p_product)) not in train and (int(userid), int(p_product), 1) not in result:
            #         result.add((int(userid), int(p_product), 0))
            #         break
            for _ in range(self.num_negative):
                item = random.randint(0, self.itemNum - 1)
                while (int(userid), item) in train:
                    item = random.randint(0, self.itemNum - 1)
                triple.add((int(userid), int(item), 0))
        print('train set num is:' + str(len(triple)))
        return list(triple), train

    def read_test(self, fname):
        print('loading test')
        for line in open(fname):
            uid, iid, label = line.strip().split()
            if int(label) == 1:
                self.test_answer.append(iid)
            if int(uid) not in self.test:
                self.test[int(uid)] = set()
            self.test[int(uid)].add((int(iid), int(label)))

    def get_test_batch(self, uid):
        x = []
        y = []
        for item, label in self.test[uid]:
            x.append((uid, item))
            y.append(label)

        return x, y


if __name__ == '__main__':
    m = DataUtil('ml_train', 6040, 3706)
    print(m.item_rated_by_user[11])
