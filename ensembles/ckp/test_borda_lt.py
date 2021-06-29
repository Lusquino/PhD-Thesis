import borda
import numpy
import wisardpkg as wp
import time
import statistics as st
from sklearn.metrics import accuracy_score

ds_train = wp.DataSet("../../datasets/ckp/lt_train.wpkds")
ds_test = wp.DataSet("../../datasets/ckp/lt_test.wpkds")

writer = open("test_borda_lt_test.txt", "w+")

y_test = []
for i in range(len(ds_test)):
    y_test.append(ds_test.getLabel(i))

'''for learners in range(10, 30, 10):
    for partitions in [0.6, 0.8]:
        for voting in ["borda0", "borda1", "dowdall"]:'''
for learners in range(20, 21):
    for partitions in [0.6]:
        for voting in ["borda0"]:
            total_training_time = []
            total_test_time = []
            total_accuracy = []
            for i in range(3):
                ensemble = borda.BordaBagging(ds_train, learners, partitions, voting)
                total_training_time.append(ensemble.get_training_time())
                test_time = time.time()
                out = ensemble.classify(ds_test)
                test_time = time.time() - test_time
                total_test_time.append(test_time)
                acc = accuracy_score(y_test, out)
                total_accuracy.append(acc)
            '''writer.write(voting + ", " + str(learners) + ", " + str(partitions) + ", " + 
            str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
            str(st.variance(total_training_time)) + "," + str(st.mean(total_test_time)) + "," +
            str(st.stdev(total_test_time)) + ", " + str(st.variance(total_test_time)) + ", " +
            str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + ", " + 
            str(st.variance(total_accuracy)) + "\n")'''

            writer.write(str(st.mean(total_test_time)) + "," +
            str(st.stdev(total_test_time)))

writer.close()