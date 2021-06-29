import kdd
from wisardpkg import DataSet, ClusRegressionWisard, PowerMean, HarmonicMean, HarmonicPowerMean, GeometricMean, ExponentialMean, SimpleMean, Median
import time

files = open("crew-test.txt", "w+")

ds_train = DataSet()

ds = DataSet("DS_train.wpkds")

for i in range(0, len(ds)):
    ds_train.add(ds.get(i), ds.getY(i))

ds_test = DataSet("DS_test.wpkds")

for i in range(1, 11):
    for address in range(5, 32):
        for  mean in [PowerMean(2), HarmonicMean(), HarmonicPowerMean(2), GeometricMean(), ExponentialMean(), SimpleMean(), Median()]:
            t_train = time.time()
            rew = ClusRegressionWisard(address, mean = mean, minScore = 0.1, threshold = 100, limit = 6)
            rew.train(ds_train)
            t_train = t_train - time.time()
            t_test = time.time()
            out = rew.predict(ds_test)
            t_test = t_test - time.time()
            kdd.create_output_file(out, "crew")
            files.write("address: " + str(address) + "; mean: " + str(mean) + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()

