import kdd
from wisardpkg import DataSet
from bagging import Regression_Bagging
import time

files = open("bagging-test.txt", "w+")

ds_train = DataSet()

ds = DataSet("DS_train.wpkds")

for i in range(0, len(ds)):
    ds_train.add(ds.get(i), ds.getY(i))

ds_test = DataSet("DS_test.wpkds")

for learners in range(100, 1100, 100):
    for i in range(1, 11):
        t_train = time.time()
        ensemble = Regression_Bagging(ds_train, learners)
        ensemble.ensemble()
        t_train = t_train - time.time()
        t_test = time.time()
        out = ensemble.predict(ds_test)
        t_test = t_test - time.time()
        kdd.create_output_file(out, "bagging_"+str(learners))
        files.write("round: " + str(i) + " - learners: " + str(learners) + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()
