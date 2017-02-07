import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def MLP(I, W, B): # inputs, weights, biases

    H = tf.nn.relu(tf.matmul(I, W[0])+B[0])
    O = tf.matmul(H, W[1])+B[1]

    return O

def MLP_var(shape):

    return tf.Variable(tf.random_normal(shape, 0, 0.1))

def do(data, get_curve_params, shuffle, plot1, plot2):

    h = 64 # nodes in hidden layer
    train_size = 65 # training size
    test_size = 13 # testing size
    data = shuffle(data)
    data = [el for el in data if (np.abs(get_curve_params(el)) <= 1e3).all() \
                   and len(get_curve_params(el)) == 3 \
                   and len(el) > 5 ]
    print(len(data))
    dataX = np.array([el[1] for el in data])
    datay = np.array([get_curve_params(el) for el in data])
    
    print(np.array(dataX).shape)
    print(np.array(datay).shape)
    train_dataX, train_datay = dataX[:train_size], datay[:train_size]
    test_dataX, test_datay = dataX[train_size:], datay[train_size:]
    epochs = 4000
    G = tf.Graph()

    with G.as_default():

        alpha = 0.5 # learning rate
 
        tr_x = tf.placeholder(tf.float32, [None, 2])
        tr_y = tf.placeholder(tf.float32, [None, 3])
        te_x = tf.placeholder(tf.float32, [None, 2])
        W = [MLP_var([2, h]), MLP_var([h, 3])]
        B = [MLP_var([h]), MLP_var([3])]
        
        predictions = MLP(tr_x, W, B)
        E = tf.reduce_mean(tf.square(predictions-tr_y))
        opt = tf.train.AdamOptimizer(alpha).minimize(E)

        pred_te_y = MLP(te_x, W, B)

    with tf.Session(graph=G) as session:

        session.run(tf.global_variables_initializer())

        for j in range(epochs):
            perm = np.random.permutation(train_size)
            train_dataX = train_dataX[perm]
            train_datay = train_datay[perm]
            nn = train_size/5
            for bi in range(5):
                feed = {tr_x: train_dataX[bi*nn:bi*nn+nn], tr_y: train_datay[bi*nn:bi*nn+nn]}
                _, e, p = session.run([opt, E, predictions], feed_dict = feed)
                if np.isnan(e):
                    print("got nan ... ")
                    #print(feed)
                    #exit(1)
            if j%500 == 0:
              print("Epoch %d: error = %f" % (j, e))

        ri = np.random.randint(0, test_size)
        while len(data[ri]) <= 5:
          ri = np.random.randint(0, test_size)
        random_case = test_dataX[ri]
        curve_params = test_datay[ri]
        x_max = data[train_size+ri][-1][0]
        print(x_max)
        print(curve_params)
        predicted_params = pred_te_y.eval(feed_dict = {te_x: [random_case]})[0]
        print(predicted_params)
        plot1(data[ri])
        plot2(predicted_params, x_max, 'r--')
        plot2(curve_params, x_max, 'g--')
        plt.show()
