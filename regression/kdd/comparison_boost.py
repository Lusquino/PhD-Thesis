# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:22:47 2019

@author: leopoldolusquino
"""

import kdd
from wisardpkg import DataSet
from boosting import Regression_Boost
import time

files = open("comparison_boost-test.txt", "w+")

ds_train = DataSet()
ds_val = DataSet()

ds_base = DataSet("DS_train.wpkds")
for i in range(0, int(9*len(ds_base)/10)):
    ds_train.add(ds_base.get(i), ds_base.getY(i))
for i in range(int(9*len(ds_base)/10), len(ds_base)):
    ds_val.add(ds_base.get(i), ds_base.getY(i))

ds_test = DataSet("DS_test.wpkds")

for i in range(1, 11):
    for model in ["rew", "crew", "heterogeneous"]:
        t_train = time.time()
        ensemble = Regression_Boost(ds_train, ds_val, 500, model)
        ensemble.ensemble()
        t_train = t_train - time.time()
        t_test = time.time()
        out = ensemble.predict(ds_test)
        t_test = t_test - time.time()
        kdd.create_output_file(out, "comparison_boost_"+str(i))
        files.write("model: " + model + "; training time: " + str(t_train) + "; test time: " + str(t_test) + "\n")

files.close()
