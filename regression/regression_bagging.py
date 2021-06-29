# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:30:25 2019

@author: leopoldolusquino
"""
import wisardpkg as wp
import random
import numpy as np
import statistics
import scipy.stats

LOW_N = 2
HIGH_N = 31
MIN_SCORE = 0.1
GROW_INTERVAL = 100
DISCRIMINATOR_LIMIT = 5

class Regression_Bagging(object):
    
    def __init__(self, train_dataset, learners, partitions = "undefined", models = "heterogeneous"):
        self.train_dataset = train_dataset
        self.learners = learners
        self.nets = []
        self.partitions = partitions
        if(partitions == "undefined"):
            self.partitions = 0.75
        if(self.partitions == 0):
            self.partitions = 0.1
        self.entry_size = len(train_dataset.get(0))
        self.models = models
        
    @staticmethod
    def random_mean():
        mean = np.random.randint(0, 7)
        if(mean == 1):
            return wp.SimpleMean()
        if(mean == 2):
            return wp.Median()
        if(mean == 3):
            return wp.HarmonicMean()
        if(mean == 4):
            return wp.HarmonicPowerMean(2)
        if(mean == 5):
            return wp.GeometricMean()
        if(mean == 6):
            return wp.ExponentialMean()
        return wp.SimpleMean()
        
    def random_model(self):
        #print(self.entry_size)
        if (random.randint(0,1))%2==0:
            add = np.random.randint(LOW_N, self.entry_size)
            net = wp.RegressionWisard(add, min_zero = np.random.randint(0, int(add/2)), 
                                           min_one = np.random.randint(0, int(add/2)), mean = self.random_mean())
        else:
            add = np.random.randint(LOW_N, self.entry_size)
            net = wp.ClusRegressionWisard(np.random.randint(LOW_N, self.entry_size), MIN_SCORE, GROW_INTERVAL, DISCRIMINATOR_LIMIT, 
                                           min_zero = np.random.randint(0, int(add/2)), min_one = np.random.randint(0, int(add/2)), 
                                           mean = self.random_mean())
        return net
    
    def random_rew(self):
        add = np.random.randint(LOW_N, self.entry_size)
        net = wp.RegressionWisard(add, min_zero = np.random.randint(0, int(add/2)), 
                                       min_one = np.random.randint(0, int(add/2)), mean = self.random_mean())
        return net
    
    def random_crew(self):
        add = np.random.randint(LOW_N, self.entry_size)
        net = wp.ClusRegressionWisard(np.random.randint(LOW_N, self.entry_size), MIN_SCORE, GROW_INTERVAL, DISCRIMINATOR_LIMIT, 
                                       min_zero = np.random.randint(0, int(add/2)), min_one = np.random.randint(0, int(add/2)), 
                                       mean = self.random_mean())
        return net
        
    def generate_dataset(self):
        data_positions = random.sample(range(0, len(self.train_dataset)), int(len(self.train_dataset)*self.partitions))
        dataset = wp.DataSet()
        for i in range(0, len(data_positions)):
            dataset.add(self.train_dataset.get(data_positions[i]), self.train_dataset.getY(data_positions[i]))
        return dataset
         
    def ensemble(self):
        for i in range(0, self.learners):
            if(self.models == "rew"):
                net = self.random_rew()
            else:
                if(self.models == "crew"):
                    net = self.random_crew()
                else:
                    net = self.random_model()
            net.train(self.generate_dataset())
            self.nets.append(net)
    
    '''@staticmethod
    def gmean(output_rams):
        gmean = 1
        counter = 0
    
        for i in range(0, len(output_rams)):
            if(not(output_rams[i][0]==0)):
                amean = output_rams[i][1]/output_rams[i][0]
                
                if(not(amean == 0)):
                    gmean *= amean
                    counter += 1

        if(counter>0):
            return pow(gmean, 1.0/counter)

        return 0'''

    @staticmethod
    def gmean(iterable):
        a = np.log(iterable)
        return np.exp(a.sum()/len(a))

    @staticmethod
    def hmean(output_rams):
        div = 0
        for i in range(0, len(output_rams)):
            if(not(output_rams[i] == 0)):
                div += 1/(output_rams[i])
        
        if(div == 0):
          return 0

        return len(output_rams)/div
    
    def predict(self, test_dataset, mean = "mean"):
        results = []
        for i in range(0, len(test_dataset)):
            result = []
            for j in range(0, len(self.nets)):
                test = wp.DataSet()
                bi = wp.BinInput(test_dataset.get(i))
                test.add(bi, 0)
                result.append(self.nets[j].predict(test)[0])
            
            result = np.ndarray.tolist(np.nan_to_num(np.array(result)))
            
            if(mean == "mean"):
                results.append(sum(result)/len(result))
            if(mean == "median"):
                results.append(statistics.median(result))
            if(mean == "harmonic"):
                results.append(self.hmean(result))
            if(mean == "geometric"):
                #results.append(scipy.stats.gmean(result))
                results.append(self.gmean(result))
                
        results = np.ndarray.tolist(np.nan_to_num(np.array(results)))
        return results
