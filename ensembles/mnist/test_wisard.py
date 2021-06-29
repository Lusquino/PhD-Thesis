import numpy
import wisardpkg as wp
import statistics as st
import time
from sklearn.metrics import accuracy_score

ds_train = wp.DataSet("../../datasets/mnist/emnist_mnist_svwd_train.wpkds")
ds_test = wp.DataSet("../../datasets/mnist/emnist_mnist_svwd_test.wpkds")

writer = open("test_wisard_test.txt", "w+")

y_test = []
for i in range(len(ds_test)):
    y_test.append(ds_test.getLabel(i))

#for address_size in range(5, 32):
for address_size in range(30, 31):
    total_training_time = []
    total_test_time = []
    total_accuracy = []
    for i in range(3):
        net = wp.Wisard(address_size)
        training_time = time.time()
        net.train(ds_train)
        training_time = time.time() - training_time
        total_training_time.append(training_time)
        test_time = time.time()
        out = net.classify(ds_test)
        test_time = time.time() - test_time
        total_test_time.append(test_time)
        acc = accuracy_score(y_test, out)
        total_accuracy.append(acc)
    '''writer.write(str(address_size) + ", " + 
    str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
    str(st.variance(total_training_time)) + "," + str(st.mean(total_test_time)) + "," +
    str(st.stdev(total_test_time)) + ", " + str(st.variance(total_test_time)) + ", " +
    str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + ", " + 
    str(st.variance(total_accuracy)) + "\n")'''
    writer.write(str(st.mean(total_test_time)) + "," +
            str(st.stdev(total_test_time)))

writer.close()