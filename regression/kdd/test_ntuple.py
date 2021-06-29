import kdd
from wisardpkg import DataSet, RegressionWisard, SimpleMean
import time

files = open("ntuple-test.txt", "w+")

ds_train = DataSet()

ds = DataSet("DS_train.wpkds")

for i in range(0, len(ds)):
    ds_train.add(ds.get(i), ds.getY(i))

ds_test = DataSet("DS_test.wpkds")

for i in range(1, 11):
    for address in range(5, 32):
        t_train = time.time()
        rew = RegressionWisard(address, mean = SimpleMean(), orderedMapping=True, completeAddressing=True)
        rew.train(ds_train)
        t_train = t_train - time.time()
        t_test = time.time()
        out = rew.predict(ds_test)
        t_test = t_test - time.time()
        kdd.create_output_file(out, "ntuple")
        files.write("address: " + str(address) + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()

