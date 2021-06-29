import matplotlib.pyplot as plt
import os
import time
import numpy as np

import util
import wisardpkg as wp
from cross_validation import (BinaryRelevance, ClusBinaryRelevance,
                              ClusLabelPowerset, CrossValidation,
                              LabelPowerset, Metrics)
from ckp import CKP, AdaptativeGaussian, AdaptativeMean, Sauvola

output_folder = "output_clus_final3/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

minScore = 0.01
threshold = 75
discriminatorLimit = 6

out = open("debug_test.txt", "w+")


#for classification_method in [BinaryRelevance, LabelPowerset, ClusBinaryRelevance, ClusLabelPowerset]:
#for classification_method in [LabelPowerset, ClusBinaryRelevance, ClusLabelPowerset]:
for classification_method in [ClusBinaryRelevance]:
    #for binary_method in [AdaptativeMean, AdaptativeGaussian, Sauvola]:
    for binary_method in [AdaptativeMean]:
        dataset = CKP(directory="dataset/data/", imgs_dir="dataset/data/CKP/cohn-kanade-images/",
        landmarks_dir="dataset/data/CKP/Landmarks", preprocessing=binary_method())
        #dataset = CKP(directory="../cohn-kanade-images/", preprocessing=binary_method()) 
        #print(dataset)

        #for addr_size in range(6, 7, 2):
        for addr_size in range(28, 31, 2):
        #for addr_size in range(6, 18, 2):
            for k in [10]:
                #print(dataset.data)
                cv = CrossValidation(dataset.data, dataset.labels, k=k)
                if((classification_method == ClusBinaryRelevance) or (classification_method == ClusLabelPowerset)):
                    result, matrix = cv.validation(method=classification_method(
                          addr_size=addr_size, minScore = minScore, threshold= threshold, discriminatorLimit = discriminatorLimit), metrics=Metrics(dataset.labels))
                else:
                    result, matrix = cv.validation(method=classification_method(
                          addr_size=addr_size), metrics=Metrics(labels = dataset.labels))
                # result = cv.validation(method=classification_method(addr_size=addr_size, minScore=minScore,
                #                                                     threshold=threshold, discriminatorLimit=discriminatorLimit), metrics=Metrics(average="macro"))

                print(result)
                labels = [item for sublist in dataset.labels for item in sublist]
                labels = list(set(labels))
                print(labels)
                result = dict(result)
                result["classification_method"] = classification_method.__name__
                result["binary_method"] = binary_method.__name__
                result["addr_size"] = addr_size
                result["k"] = k
                print(result["f1_score"])
               # plt.figure(figsize=(15, 3))  # width:20, height:3
               # plt.bar(cv.mlb.classes_,
                #        result["f1_score"], align='center', width=0.8)
                #plt.show()

                util.dictToJSON(output_folder + time.strftime(
                     "%Y%m%d-%H%M%S") + ".json", result)
                print("-------------")
                out.write(str(k) + "\n")

out.close()
