"""
Created on Thu Jun 13 21:23:50 2019

@author: leopoldolusquino
"""

import kdd
from wisardpkg import DataSet
from boosting import Regression_Boost
import time

files = open("boost-test.txt", "w+")

ds_train = DataSet()
ds_val = DataSet()

ds_base = DataSet("DS_train.wpkds")
for i in range(0, int(9*len(ds_base)/10)):
    ds_train.add(ds_base.get(i), ds_base.getY(i))
for i in range(int(9*len(ds_base)/10), len(ds_base)):
    ds_val.add(ds_base.get(i), ds_base.getY(i))

ds_test = DataSet("DS_test.wpkds")

for learners in range(100, 1100, 100):
    for i in range(1, 11):
        t_train = time.time()
        ensemble = Regression_Boost(ds_train, ds_val, learners)
        ensemble.ensemble()
        t_train = t_train - time.time()
        t_test = time.time()
        out = ensemble.predict(ds_test)
        t_test = t_test - time.time()
        kdd.create_output_file(out, "boost_"+str(learners))
        files.write("round: " + str(i) + "- learners: " + str(learners) + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()
