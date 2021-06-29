import wisardpkg as wp 

ds = wp.DataSet("../../datasets/cifar_simple/simple_cifar_train_10.wpkds")

print(len(ds))

labels = {}

for i in range(len(ds)):
    if(ds.getLabel(i) in labels):
        labels[ds.getLabel(i)] = labels[ds.getLabel(i)] + 1
    else:
        #labels.append(ds.getLabel(i))
        labels[ds.getLabel(i)] = 0
    #print(ds.getLabel(i))
print(labels)