import tensorflow as tf
from DataUtil2 import DataUtil

if __name__=='__main__':
    d = DataUtil('ml_train', 6040, 3706)
    max = 0
    index = 0
    for i in range(6040):
        rated_set = d.item_rated_by_user[i]
        num = len(rated_set)
        if num > max:
            max = num
            index = i
    print(max)
    print(index)
