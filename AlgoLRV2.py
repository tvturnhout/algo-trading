import Quandl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import tensorflow as tf
from datetime import datetime
import pickle
rng = np.random

start_date="2015-01-01"
end_date="2016-01-01"
days_to_look_back=5


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def linreg(Yin):

    #X-axis create lineair vector with same length as  Y
    n_samples = len(Yin)
    Xin=np.arange(1,n_samples+1).tolist()

    #Make tf variables
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Parameters
    learning_rate = 0.01
    training_epochs = 100
    display_step = 25

    #Creating random weights and biases
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    #Create a linear model of the form Y=W.X+b
    activation = tf.add(tf.mul(X, W), b)

    #Define objective - in this case linear regression it is minimizing the squared errors
    cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
    optimizer = tf.train.GradientDescentOptimizer(max(learning_rate*cost,learning_rate)).minimize(cost) #Gradient descent

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the session
    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(Xin, Yin):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            #Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: Xin, Y:Yin})), \
                    "W=", sess.run(W), "b=", sess.run(b)

        print "Optimization Finished!"
        training_cost = sess.run(cost, feed_dict={X: Xin, Y: Yin})
        print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'
        Wo=sess.run(W)
        bo=sess.run(b)


    #Plotterino
    #Xplot=np.asarray(Xin)
    #Yplot=np.asarray(Yin)
    #plt.plot(Xplot, Yplot, 'ro', label='Original data')
    #plt.plot(Xplot, Wo * Xplot + bo, label='Fitted line')
    #plt.legend()
    #plt.show()

    #Make prediction
    Yfuture=(len(Xin)+1)*Wo+bo
    return Yfuture

EURUSD = Quandl.get("ECB/EURUSD",trim_start=start_date,trim_end=end_date, authtoken="26ckMP3atyV1AC41LU3z")
EURUSDlist = EURUSD['Value'].values.tolist()

backtest_iterations = int(round(days_between(start_date,end_date)/days_to_look_back))
modulo = days_between(start_date,end_date)%days_to_look_back

Youtlist=[]
for i in range(days_between(start_date,end_date)-modulo):
    print "Backtesting day", i, "of", days_between(start_date,end_date)-modulo
    Yinput = EURUSDlist[i+modulo:i+modulo+days_to_look_back]
    Youtlist.append(linreg(Yinput))

original=EURUSDlist[-len(Youtlist):]
prediction=Youtlist

f = open("original.txt", "w")
f.write("\n".join(map(lambda x: str(x), original)))
f.close()

f = open("prediction.txt", "w")
f.write("\n".join(map(lambda x: str(x), prediction)))
f.close()

