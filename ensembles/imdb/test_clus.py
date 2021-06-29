import numpy
import wisardpkg as wp
import statistics as st
import time
from sklearn.metrics import accuracy_score

MIN_SCORE = 0.1
GROW_INTERVAL = 100

ds_train = wp.DataSet("../../datasets/imdb/th_train.wpkds")
ds_test = wp.DataSet("../../datasets/imdb/th_test.wpkds")

writer = open("test_clus_test.txt", "w+")

y_test = []
for i in range(len(ds_test)):
    y_test.append(ds_test.getLabel(i))

'''for address_size in range(5, 32):
    for discriminator_limit in range(3, 6):'''
for address_size in range(5, 6):
    for discriminator_limit in range(4, 5):
        total_training_time = []
        total_test_time = []
        total_accuracy = []
        for i in range(10):
            net = wp.ClusWisard(address_size, MIN_SCORE, GROW_INTERVAL, discriminator_limit)
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
        '''writer.write(str(address_size) + ", " + str(discriminator_limit) + ", " +
        str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
        str(st.variance(total_training_time)) + "," + str(st.mean(total_test_time)) + "," +
        str(st.stdev(total_test_time)) + ", " + str(st.variance(total_test_time)) + ", " +
        str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + ", " + 
        str(st.variance(total_accuracy)) + "\n")'''
        writer.write(str(st.mean(total_test_time)) + "," +
            str(st.stdev(total_test_time)))
        
writer.close()