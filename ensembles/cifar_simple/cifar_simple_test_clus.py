import numpy
import wisardpkg as wp
import statistics as st
import time
from sklearn.metrics import accuracy_score

MIN_SCORE = 0.1
GROW_INTERVAL = 100

#bits = ["10", "15", "20"]
bits = ["10", "15"]

for bit in bits:
    ds_train = wp.DataSet("../../datasets/cifar_simple/simple_cifar_train_" + bit + ".wpkds")
    ds_test = wp.DataSet("../../datasets/cifar_simple/simple_cifar_test_" + bit + ".wpkds")

    writer = open("test_clus_test_" + bit + ".txt", "w+")

    y_test = []
    for i in range(len(ds_test)):
        y_test.append(ds_test.getLabel(i))

    #for address_size in [5, 10, 15, 20, 25, 30]:
    for address_size in [7]:
        #for discriminator_limit in range(3, 6):
        for discriminator_limit in [3]:
            total_training_time = []
            total_test_time = []
            total_accuracy = []
            for i in range(3):
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
            print("FOI: " + str(address_size))
            writer.write(str(address_size) + ", " + str(discriminator_limit) + ", " +
            str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
            str(st.mean(total_test_time)) + "," + str(st.stdev(total_test_time)) + ", " +
            str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy))  + "\n")
            
    writer.close()