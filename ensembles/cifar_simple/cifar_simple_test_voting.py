import voting
import numpy
import wisardpkg as wp
import statistics as st
import time
from sklearn.metrics import accuracy_score

#bits = ["10", "15", "20"]
bits = ["10", "15"]

for bit in bits:
    ds_train = wp.DataSet("../../datasets/cifar_simple/simple_cifar_train_" + bit + ".wpkds")
    ds_test = wp.DataSet("../../datasets/cifar_simple/simple_cifar_test_" + bit + ".wpkds")

    writer = open("test_wv_test_" + bit + ".txt", "w+")

    y_test = []
    for i in range(len(ds_test)):
        y_test.append(ds_test.getLabel(i))

    for learners in [20]:
        for partitions in [0.6]:
            #for vote in ["plurality1", "plurality2", "plurality3", "plurality4"]:
            for vote in ["plurality3"]:
                total_training_time = []
                total_test_time = []
                total_accuracy = []
                for i in range(3):
                    ensemble = voting.VotingBagging(ds_train, learners, partitions, vote)
                    total_training_time.append(ensemble.get_training_time())
                    test_time = time.time()
                    out = ensemble.classify(ds_test)
                    if(None in out):
                        for i in range(len(out)):
                            if(out[i] == None):
                                out[i] = y_test[0]
                    test_time = time.time() - test_time
                    total_test_time.append(test_time)
                    acc = accuracy_score(y_test, out)
                    total_accuracy.append(acc)
                    print("ENSEMBLOU")
                writer.write(vote + ", " + str(learners) + ", " + str(partitions) + ", " + 
                str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," + 
                str(st.mean(total_test_time)) + "," + str(st.stdev(total_test_time)) + ", " +
                str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + "\n")

    writer.close()