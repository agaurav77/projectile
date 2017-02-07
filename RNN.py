#!/bin/python2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class ProjectileData:

    def __init__(self):
        data = np.loadtxt(open("projectiles.csv", "rb"), delimiter=",").astype(np.float32)
        #examples = np.array(self.separate_projectiles(data, delete_time_axis=False))
        #self.examples = self.shuffle(examples)       
        #self.examples_no_time = self.delete_time(self.examples)
        self.orig_examples = np.array(self.separate_projectiles(data))
        self.examples = self.shuffle(self.orig_examples)
        self.examples_no_time = self.examples # a bit misleading
        self.max_points = 50
        self.batch_size = 25
        self.curr_batch = 0
        self.test_batch = 0
        # with a batch_size of 25, 1325 examples = 53 batches
        # use 50 batches for training, 3 for testing

        self.X = []
        self.y = []
        self.seqlens = []

        # pts are [1, 2, 3]
        # [1] -> [2]
        # [1, 2] -> [3]
        n_examples = len(self.examples)
        for e_idx in range(n_examples):
            e_len = len(self.examples[e_idx])
            #local_X, local_y = self.convert_example(example)
            #self.X.append(self.pad_example(local_X, self.max_points-e_len, pts = 3))
            #self.y.append(self.pad_example(local_y, self.max_points-e_len))
            for i in range(1, e_len):
                self.X.append(self.pad_example(self.examples[e_idx][:i], self.max_points-i))
                self.y.append(self.examples_no_time[e_idx][i])
                self.seqlens.append(i)
        print("Total Examples = %d" % len(self.X))
        self.num_batches_train = len(self.X)/self.batch_size - 3
        self.num_batches_test = 3

    def delete_time(self, X):
        ret = []
        for example in X:
            ret2 = []
            for pt in example:
                ret2.append(np.delete(pt, 0))
            ret.append(ret2)
        return np.array(ret)


    # return X, y, seqlens for next batch of examples (training)
    def next_batch(self):
        bid = self.curr_batch
        new_bid = (bid+1)%self.num_batches_train
        bN = self.batch_size
        if new_bid == 0:
            bX = self.X[bid*bN:]
            by = self.y[bid*bN:]
            bseqlens = self.seqlens[bid*bN:]
        else:
            bX = self.X[bid*bN:new_bid*bN]
            by = self.y[bid*bN:new_bid*bN]
            bseqlens = self.seqlens[bid*bN:new_bid*bN]
        self.curr_batch = new_bid
        return bX, by, bseqlens

    def next_test_batch(self):
        bid = self.test_batch
        new_bid = (bid+1)%self.num_batches_test
        bN = self.batch_size
        before = self.num_batches_train*bN # training examples overlooked
        if new_bid == 0:
            bX = self.X[before+bid*bN:]
            by = self.y[before+bid*bN:]
            bseqlens = self.seqlens[before+bid*bN:]
        else:
            bX = self.X[before+bid*bN:before+new_bid*bN]
            by = self.y[before+bid*bN:before+new_bid*bN]
            bseqlens = self.seqlens[before+bid*bN:before+new_bid*bN]
        self.test_batch = new_bid
        return bX, by, bseqlens


    # 100 x points x 3 (where points <= 50)
    def separate_projectiles(self, data, delete_time_axis=True):
        start = [0, 0, 0]
        if delete_time_axis:
            data = np.delete(data, 0, axis=1) # delete col 1
            start = [0, 0]
        n = len(data)
        res = []
        for i in range(n):
            curr = data[i]
            if (curr == start).all():
                res.append([curr])
            else:
                res[-1].append(curr)
        return np.array(res)

    # shuffle data
    def shuffle(self, d):
        return np.random.permutation(d)

    # convert example to training form
    # so [(1,3,3), (2,4,4), (3,5,5)]
    # becomes X = [(1,3,3), (2,4,4)] and y = [(4,4), (5,5)]
    def convert_example(self, example):
        #example_no_time = np.delete(example, 0, axis=1)
        #example2 = [ent for ent in example_no_time]
        return example[:-1], example[1:]

    # pad with dummy points (will not learn from these points)
    def pad_example(self, x, extra, pts = 2):
        local_n = len(x)
        if pts == 3:
            paddings = [np.array([0., 0., 0.]) for i in range(extra)]
        else:
            paddings = [np.array([0., 0.]) for i in range(extra)]
        tmp_x = x[:]
        tmp_x.extend(paddings)
        return tmp_x

    def format_X(self, X):
        strs = [element.__str__() for element in X]
        return '\n'.join(strs)


class RNNModel:

    def __init__(self):

        self.alpha = 0.01
        self.num_epochs = 20000
        self.nH = 64 # hidden layers
        self.nO = 2 # outputs
        self.batch_size = 25

        self.dataset = ProjectileData()
        #self.test_set = ProjectileData()

        self.construct_graph_and_train()

    # dynamic RNN
    # def dRNN(self, passedX, seqlen, W, B):

    #     # seq_index, batch_index, coordinates
    #     passedX = tf.transpose(passedX, [1, 0, 2])
    #     # seq_index*batch_index, coordinates
    #     passedX = tf.reshape(passedX, [-1, 2])
    #     # get 50 tensors of shape batch_index, coordinates
    #     passedX = tf.split(0, 50, passedX)

    #     self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.nH)

    #     self.outputs, self.states = tf.nn.rnn(self.cell, passedX, dtype=tf.float32, sequence_length=seqlen)

    #     self.outputs = tf.pack(self.outputs)
    #     # batch_index, seq_index, coordinates
    #     self.outputs = tf.transpose(self.outputs, [1, 0, 2])

    #     self.indices = tf.range(0, self.batch_size)*50 + (seqlen - 1)
    #     self.outputs = tf.gather(tf.reshape(self.outputs, [-1, self.nH]), self.indices)

    #     return tf.matmul(self.outputs, W) + B


    # create RNN graph
    def construct_graph_and_train(self):

        G = tf.Graph()
        with G.as_default():

            # batch_index, seq_index, coordinates
            X = tf.placeholder(tf.float32, [None, 50, 2])
            # batch_index, output_coordinates
            y = tf.placeholder(tf.float32, [None, 2])
            # batch_index
            seqlen = tf.placeholder(tf.int32, [None])

            W = tf.Variable(tf.random_normal([self.nH, self.nO]))
            B = tf.Variable(tf.random_normal([self.nO]))

            #pred = self.dRNN(X, lens, W, B)

            # seq_index, batch_index, coordinates
            X1 = tf.transpose(X, [1, 0, 2])
            # seq_index*batch_index, coordinates
            X1 = tf.reshape(X1, [-1, 2])
            # get 50 tensors of shape batch_index, coordinates
            X1 = tf.split(0, 50, X1)

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.nH)

            outputs, states = tf.nn.rnn(cell, X1, dtype=tf.float32, sequence_length=seqlen)

            outputs = tf.pack(outputs)
            # batch_index, seq_index, coordinates
            outputs = tf.transpose(outputs, [1, 0, 2])

            batch_size2 = tf.shape(outputs)[0]
            indices = tf.range(0, batch_size2)*50 + (seqlen - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.nH]), indices)

            pred = tf.matmul(outputs, W) + B

            E = tf.reduce_mean(tf.square(pred-y))
            opt = tf.train.GradientDescentOptimizer(self.alpha).minimize(E)

        with tf.Session(graph = G) as session:

            session.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):
                bX, by, bseqlens = self.dataset.next_batch()
                bX = np.array(bX)
                by = np.array(by)
                bseqlens = np.array(bseqlens)
                session.run(opt, feed_dict = {
                    X: bX, 
                    y: by, 
                    seqlen: bseqlens
                })
                bE = session.run(E, feed_dict = {X: bX, y: by, seqlen: bseqlens})

                print("Epoch %d: E = %f" % (epoch, bE))

            print("Optimization complete!")

            for batch_idx in range(3):

                print("Testing (batch %d)" % batch_idx)

                bX, by, bseqlens = self.dataset.next_test_batch()
                bX = np.array(bX)
                by = np.array(by)
                bseqlens = np.array(bseqlens)
                test_pred = session.run(pred, feed_dict = {
                    X: bX, 
                    y: by, 
                    seqlen: bseqlens
                })

                n_test = len(test_pred)
                for i in range(n_test):
                    print("Test %d:" % i)
                    #print("X = ", bX[i][:bseqlens[i]])
                    #print("y = ", by[i])
                    #print("predicted = ", test_pred[i])
                    print(self.dataset.format_X(bX[i][:bseqlens[i]])+" -> "+by[i].__str__()+"\npredicted = "+test_pred[i].__str__()+"\n")


            # Lets see one of these predictions
            ex = self.dataset.examples[0]
            e_len = len(ex)
            tmpX = []
            tmpy = []
            tmpseqlens = []
            pred_pts = [ex[0], ex[1]] # except the first 2 pts
            for i in range(2, e_len):
                tmpX.append(self.dataset.pad_example(pred_pts[:i], 50-i))
                tmpy.append(ex[i]) # bogus
                tmpseqlens.append(i)
                test_pred = session.run(pred, feed_dict = {
                    X: np.array(tmpX), 
                    y: np.array(tmpy), 
                    seqlen: np.array(tmpseqlens)
                })
                pred_pts.append(test_pred[-1])
            print("actual example : ")
            print(ex)
            print("predicted : ")
            print(pred_pts)

            np_points1 = np.array(ex)
            plt.plot(np_points1[:, 0], np_points1[:, 1], linewidth=1.4)
            np_points2 = np.array(pred_pts)
            plt.plot(np_points2[:, 0], np_points2[:, 1], 'b--')
            plt.show()


if __name__ == "__main__":

    rnn = RNNModel()
    #rnn.train()