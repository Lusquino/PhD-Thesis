import boost
import wisardpkg as wp
import time
import statistics as st
from sklearn.metrics import accuracy_score

bits = ["10", "15"]
#bits = ["10", "15", "20"]

for bit in bits:
    ds_base = wp.DataSet("../../datasets/cifar_simple/simple_cifar_train_" + bit + ".wpkds")
    ds_test = wp.DataSet("../../datasets/cifar_simple/simple_cifar_test_" + bit + ".wpkds")

    ds_train = wp.DataSet()
    ds_val = wp.DataSet()

    for i in range(0, int(9*len(ds_base)/10)):
        ds_train.add(ds_base.get(i), ds_base.getLabel(i))
    for i in range(int(9*len(ds_base)/10), len(ds_base)):
        ds_val.add(ds_base.get(i), ds_base.getLabel(i))

    writer = open("test_boost_test_" + bit + ".txt", "w+")

    y_test = []
    for i in range(len(ds_test)):
        y_test.append(ds_test.getLabel(i))

    #for learners in range(10, 30, 10):
    for learners in [10]:
        #for models in ["wisard", "clus", "heterogeneous"]:
        for models in ["wisard"]:
            total_training_time = []
            total_validation_time = []
            total_test_time = []
            total_accuracy = []
            for i in range(3):
                ensemble = boost.Boost(ds_train, ds_val, learners, models)
                total_training_time.append(ensemble.get_training_time())
                total_validation_time.append(ensemble.get_validation_time())
                test_time = time.time()
                out = ensemble.classify(ds_test)
                test_time = time.time() - test_time
                total_test_time.append(test_time)
                acc = accuracy_score(y_test, out)
                total_accuracy.append(acc)
            writer.write(models + ", " + str(learners) + ", " + 
                str(st.mean(total_training_time)) + ", " + str(st.stdev(total_training_time)) + "," +
                str(st.mean(total_test_time)) + "," + str(st.stdev(total_test_time)) + ", " +
                str(st.mean(total_accuracy)) + ", " + str(st.stdev(total_accuracy)) + "\n")

    writer.close()