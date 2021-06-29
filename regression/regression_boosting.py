# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:03:42 2019

@author: leopoldolusquino
"""

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
from sklearn.preprocessing import normalize

LOW_N = 2
HIGH_N = 31
MIN_SCORE = 0.1
GROW_INTERVAL = 100
DISCRIMINATOR_LIMIT = 5

class Regression_Boost(object):
    
    def __init__(self, train_dataset, validation_dataset, learners, models = "heterogeneous"):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        if(learners < len(train_dataset)):
            self.learners = learners
        else:
            self.learners = len(train_dataset)
        self.models = models
        self.nets = []
        self.entry_size = len(train_dataset.get(0))
        self.data_positions = list(range(0, len(self.train_dataset)))
        self.ensemble_weights = [0] * learners
        self.total_sum = 0
        
    @staticmethod
    def random_mean():
        mean = np.random.randint(0, 7)
        if(mean == 1):
            return wp.PowerMean(2)
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
        local_data_positions = random.sample(self.data_positions, int(len(self.train_dataset)/self.learners))
        dataset = wp.DataSet()
        for i in range(0, len(local_data_positions)):
            self.data_positions.remove(local_data_positions[i])
            dataset.add(self.train_dataset.get(local_data_positions[i]), self.train_dataset.getY(local_data_positions[i]))
        return dataset    

    def validate(self, net):
        results = net.predict(self.validation_dataset)
        mae = 0
        for i in range(0, len(self.validation_dataset)):
            mae += self.validation_dataset.getY(i) - results[i]
        return 1 - mae/len(self.validation_dataset)
        
    def normalize_weights(self):
        '''total = 0
        for i in range(0, len(self.ensemble_weights)):
            total += self.ensemble_weights[i]
        self.total_sum = total'''
        
        #self.ensemble_weights = np.isnan(np.asarray(self.ensemble_weights)).tolist()       
        
        for i in range(0, len(self.ensemble_weights)):
            if(np.isnan(self.ensemble_weights[i])):
                self.ensemble_weights[i] = 0
        
        self.total_sum = sum(self.ensemble_weights)
        
        #print("TOTAL: " + str(self.total_sum))
        for i in range(0, len(self.ensemble_weights)):
            #print("PESO: " + str(self.ensemble_weights[i]))
            if(self.total_sum == 0):
                self.ensemble_weights[i] = 0
            else:
                self.ensemble_weights[i] = (self.ensemble_weights[i] * 100)/self.total_sum
                #print("VIROU: " + str(self.ensemble_weights[i]))
        
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
            self.ensemble_weights[i] = self.validate(net)
            self.nets.append(net)
        self.normalize_weights()

    ''' @staticmethod
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
    def gmean(output):
        a = np.log(output)
        return np.exp(a.sum()/len(a))

    @staticmethod
    def hmean(output_rams):
        div = 0
        
        #print("HMEAN")
        #print(len(output_rams))
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
            
            for j in range(0, len(self.nets)):
                result[j] = result[j] * self.ensemble_weights[j]
                
            #print("results")
                
            result = np.ndarray.tolist(np.nan_to_num(np.array(result)))
            
            #print("no nans")
                         
            if(mean == "mean"):
                results.append(sum(result)/len(result))
            if(mean == "median"):
                results.append(statistics.median(result))
            if(mean == "harmonic"):
                results.append(self.hmean(result))
            if(mean == "geometric"):
                #print("geometric")
                #results.append(scipy.stats.gmean(result))
                results.append(self.gmean(result))
            
            #print("PASSOU: " + str(i) + "/" + str(len(test_dataset)))
        results = np.ndarray.tolist(np.nan_to_num(np.array(results)))
        return results
