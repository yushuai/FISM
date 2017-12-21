def proc_train(dataset, fname):
    target = open('./' + dataset + '_train', 'w')
    for line in open(fname):
        uid, iid, _, _ = line.strip().split()
        target.write(uid)
        target.write('\t')
        target.write(iid)
        target.write('\n')
    target.close()


def proc_test(dataset, fname, fname_negative):
    target = open('./' + dataset + '_test', 'w')
    t2 = open(fname_negative)
    for line in open(fname):
        uid, iid, _, _ = line.strip().split()
        arr = t2.readline().strip().split()
        target.write(uid)
        target.write('\t')
        target.write(iid)
        target.write('\t')
        target.write('1')
        target.write('\n')
        for i in range(1, len(arr)):
            target.write(uid)
            target.write('\t')
            target.write(arr[i])
            target.write('\t')
            target.write('0')
            target.write('\n')

    target.close()


if __name__ == '__main__':
    proc_train('ml', '/Users/yushuai/Downloads/Neural-Attentive-Item-Similarity-Model-master/Data/ml-1m.train.rating')
    proc_test('ml', '/Users/yushuai/Downloads/Neural-Attentive-Item-Similarity-Model-master/Data/ml-1m.test.rating',
              '/Users/yushuai/Downloads/Neural-Attentive-Item-Similarity-Model-master/Data/ml-1m.test.negative')
