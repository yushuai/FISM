import numpy as np
import tensorflow as tf
from DataUtil2 import DataUtil
import copy


class Fism:
    def __init__(self):
        self.user_num = 6040
        self.item_num = 3706
        self.K = 64
        self.batch = 1024
        self.max_len = 2313
        self.lambda_reg = self.gamma_reg = 5e-3
        self.learning_rate = 8e-4
        self.P = tf.Variable(tf.random_uniform([self.item_num, self.K], minval=-0.1, maxval=0.1))
        self.Q = tf.Variable(tf.random_uniform([self.item_num, self.K], minval=-0.1, maxval=0.1))
        self.zero_vector = tf.constant(0.0,tf.float32, [1, self.K])
        self.Q = tf.concat([self.Q, self.zero_vector], 0)
        self.bias_u = tf.Variable(tf.random_uniform([self.user_num, 1], minval=-0.1, maxval=0.1))
        self.bias_i = tf.Variable(tf.random_uniform([self.item_num, 1], minval=-0.1, maxval=0.1))
        self.X = tf.placeholder(dtype=tf.int32, shape=(None, 2))
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.alpha = tf.Variable(tf.constant(0.5), dtype=tf.float32)
        self.neighbour = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        self.optimizer = None
        self.loss = None
        self.logits = None
        self.neighbour_num = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.helper = DataUtil(
            'ml_train',
            self.user_num, self.item_num)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            max_hit_ratio = 0
            for epoch in range(30):
                print('epoch ' + str(epoch))
                iteration_num = len(self.helper.train) / self.batch
                print('iteration num is:%d' % (iteration_num))
                for i in range(int(iteration_num)):
                    if i == 0:
                        x_train, y_train = self.helper.next_batch(self.batch, reset=True)
                    else:
                        x_train, y_train = self.helper.next_batch(self.batch)
                    y_train = np.reshape(np.array(y_train), (self.batch))
                    x_train_neighbours = []
                    len_neighbour = []
                    # uniform the neighbour length
                    for (uid, iid), y_label in zip(x_train, y_train):
                        rated_set = list(self.helper.item_rated_by_user[uid])
                        len_neighbour.append(len(rated_set))
                        if y_label == 1:
                            rated_set.remove(iid)
                        if len(rated_set) < self.max_len:
                            rated_set = list(rated_set)
                            while len(rated_set) < self.max_len:
                                if len(rated_set) == 0:
                                    rated_set.append(self.item_num)
                                else:
                                    rated_set.append(self.item_num)
                        rated_set = list(rated_set)
                        x_train_neighbours.append(rated_set[:self.max_len])
                    len_neighbour = np.array(len_neighbour)
                    # train batch
                    _, loss_train, l = sess.run([self.optimizer, self.loss, self.logits],
                                                feed_dict={
                                                    self.Y: y_train,
                                                    self.X: x_train,
                                                    self.neighbour_num: len_neighbour,
                                                    self.neighbour: x_train_neighbours,

                                                })
                    print('loss:%f' % (loss_train))
                    # evaluate test data set
                    if i % 100 == 0:
                        hit_ratio = self.evaluate(sess)
                        print('hit ratio is %f' % (hit_ratio))
                        if hit_ratio > max_hit_ratio:
                            max_hit_ratio = hit_ratio
                            print('saving best hit ratio:%f' % (hit_ratio))
                            tf.train.Saver().save(sess, './model_save')

    def build_graph(self):
        u_i = tf.split(self.X, 2, axis=1)
        u = tf.reshape(u_i[0], [-1])
        i = tf.reshape(u_i[1], [-1])
        item_emb = tf.nn.embedding_lookup(self.P, i)
        sumvec = tf.reduce_sum(tf.gather(self.Q, self.neighbour), 1)
        # inverse_rated_num = tf.pow(self.rated_num, -tf.constant(self.alpha, tf.float32, [1]))
        inverse_rated_num = tf.pow(self.neighbour_num, -self.alpha)
        inverse_rated_num = tf.reshape(inverse_rated_num, [-1, 1])
        user_repr = tf.multiply(inverse_rated_num, sumvec)
        self.rating = tf.reduce_sum(item_emb * user_repr, axis=1)
        bias_u = tf.reshape(tf.nn.embedding_lookup(self.bias_u, u), [-1])
        bias_i = tf.reshape(tf.nn.embedding_lookup(self.bias_i, i), [-1])
        self.logits = tf.nn.sigmoid(self.rating + bias_u + bias_i)
        self.loss = -tf.reduce_sum(
            self.Y * tf.log(self.logits + 1e-10) + (1 - self.Y) * tf.log(1 - self.logits + 1e-10)) / self.batch
        self.loss += self.lambda_reg * tf.reduce_sum(tf.square(self.P))
        self.loss += self.gamma_reg * tf.reduce_sum(tf.square(self.Q))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
        #                                           initial_accumulator_value=1e-8).minimize(self.loss)

    def evaluate(self, sess):
        hit_num = 0
        for i in range(self.user_num):
            x_test, y_test = self.helper.get_test_batch(i)
            rated_set = self.helper.item_rated_by_user[i]
            neighbour_number = len(rated_set)
            rated_set = list(rated_set)
            if len(rated_set) < self.max_len:
                while len(rated_set) < self.max_len:
                    rated_set.append(self.item_num)
            feed_neighbour = []
            for j in range(100):
                feed_neighbour.append(rated_set)
            score = sess.run(self.logits, feed_dict={
                self.Y: y_test,
                self.X: x_test,
                self.neighbour: feed_neighbour,
                self.neighbour_num: [neighbour_number for i in x_test]
            })
            score = np.array(score)
            item_score = []
            for index, t in enumerate(score):
                item_score.append((x_test[index][1], t))
            item_score.sort(key=lambda k: k[1], reverse=True)
            rec_list = set()
            for t in item_score:
                rec_list.add(t[0])
                if len(rec_list) == 10:
                    break
            answer = self.helper.test_answer[i]
            #print('answer is %s' % (answer))
            #print(rec_list)
            if int(answer) in rec_list:
                hit_num += 1
        return hit_num / self.user_num


if __name__ == '__main__':
    model = Fism()
    model.build_graph()
    model.train()
