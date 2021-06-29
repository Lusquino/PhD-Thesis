import wisard_bagging as bg
import numpy
import wisardpkg as wp
import statistics as st
import time
from sklearn.metrics import accuracy_score

ds_train = wp.DataSet("../../datasets/cifar10/mt_train.wpkds")
ds_test = wp.DataSet("../../datasets/cifar10/mt_test.wpkds")

writer = open("test_bagging_mt_test.txt", "w+")

y_test = []
for i in range(len(ds_test)):
    y_test.append(ds_test.getLabel(i))

#for learners in range(10, 30, 10):
for learners in range(20, 21):   
    #for partitions in [0.6, 0.8]:
    for partitions in [0.6]:
        #for models in ["wisard", "clus", "heterogeneous"]:
        for models in ["clus"]:
            total_training_time = []
            total_test_time = []
            total_accuracy = []
            for i in range(3):
                ensemble = bg.Bagging(ds_train, learners, partitions, models)
                total_training_time.append(ensemble.get_training_time())
                test_time = time.time()
                out = ensemble.classify(ds_test)
                test_time = time.time() - test_time
                total_test_time.append(test_time)
                acc = accuracy_score(y_test, out)
                total_accuracy.append(acc)
            #print(str(st.mean(total_test_time)) + "," +
            #str(st.stdev(total_test_time)))
            #writer.write(models + ", " + str(learners) + ", " + str(partitions) + ", " + 
           # str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
           # str(st.variance(total_training_time)) + "," + str(st.mean(total_test_time)) + "," +
           # str(st.stdev(total_test_time)) + ", " + str(st.variance(total_test_time)) + ", " +
           # str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + ", " + 
           # str(st.variance(total_accuracy)) + "\n")

            writer.write(str(st.mean(total_test_time)) + "," +
            str(st.stdev(total_test_time)))

writer.close()