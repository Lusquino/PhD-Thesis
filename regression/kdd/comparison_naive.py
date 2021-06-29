import kdd
from wisardpkg import DataSet
from bagging import Regression_Bagging
import time

files = open("comparison_naive-test.txt", "w+")

ds_train = DataSet()

ds = DataSet("DS_train.wpkds")

for i in range(0, len(ds)):
    ds_train.add(ds.get(i), ds.getY(i))

ds_test = DataSet("DS_test.wpkds")

for i in range(1, 11):
    for model in ["rew", "crew", "heterogeneous"]:
        t_train = time.time()
        ensemble = Regression_Bagging(ds_train, 500, partitions = 1, models = model)
        ensemble.ensemble()
        t_train = t_train - time.time()
        t_test = time.time()
        out = ensemble.predict(ds_test)
        t_test = t_test - time.time()
        kdd.create_output_file(out, "comparison_naive_" + str(i))
        files.write("model: " + str(model) + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()

