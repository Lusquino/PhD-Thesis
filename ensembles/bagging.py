import wisardpkg as wp
import random
import numpy as np

LOW_N = 5
HIGH_N = 31
MIN_SCORE = 0.1
GROW_INTERVAL = 100
MAX_DISCRIMINATOR_LIMIT = 10

class Bagging(object):
    
    def __init__(self, train_dataset, learners, partitions = "undefined", models = "heterogeneous"):
        self.train_dataset = train_dataset
        self.learners = learners
        self.nets = []
        self.partitions = partitions
        if(partitions == "undefined"):
            self.partitions = int(len(train_dataset)/75)
        if(self.partitions == 0):
            self.partitions = 1
        self.models = models
        self.entry_size = len(train_dataset.get(0))
        
    def random_model(self):
        if (random.randint(0,1))%2==0:
            net = wp.Wisard(np.random.randint(LOW_N, HIGH_N))
        else:
            discriminator_limit = np.random.randint(2, MAX_DISCRIMINATOR_LIMIT)
            net = wp.ClusWisard(np.random.randint(LOW_N, HIGH_N), MIN_SCORE, GROW_INTERVAL, discriminator_limit)
        return net
        
    def random_wisard(self):
        return wp.Wisard(np.random.randint(LOW_N, HIGH_N))
        
    def random_clus(self):
        discriminator_limit = np.random.randint(2, MAX_DISCRIMINATOR_LIMIT)
        return wp.ClusWisard(np.random.randint(LOW_N, HIGH_N), MIN_SCORE, GROW_INTERVAL, discriminator_limit)
    
    def generate_dataset(self):
        data_positions = random.sample(range(0, len(self.train_dataset)), int(len(self.train_dataset)*self.partitions))
        dataset = wp.DataSet()
        for i in range(0, len(data_positions)):
            dataset.add(self.train_dataset.get(data_positions[i]), self.train_dataset.getLabel(data_positions[i]))
        return dataset
         
    def ensemble(self):
        if(self.models == "heteronegous"):
            for i in range(0, self.learners):
                net = self.random_model()
                net.train(self.generate_dataset())
                self.nets.append(net)
        else:
            if(self.models == "wisard"):
                for i in range(0, self.learners):
                    net = self.random_wisard()
                    net.train(self.generate_dataset())
                    self.nets.append(net)
            else:
                for i in range(0, self.learners):
                    net = self.random_clus()
                    net.train(self.generate_dataset())
                    self.nets.append(net)
    
    def classify(self, test_dataset):
        results = []
        for i in range(0, len(test_dataset)):
            result = {}
            #print("classificando: " + str(i))
            for j in range(0, len(self.nets)):
                test = wp.DataSet()
                bi = wp.BinInput(test_dataset.get(i))
                test.add(bi, test_dataset.getLabel(i))
                r = self.nets[j].classify(test)
                
                if(r[0] in result):
                    result[r[0]] += 1
                else:
                    result[r[0]] = 1
                    
            results.append(max(result, key = result.get))
               
        return results