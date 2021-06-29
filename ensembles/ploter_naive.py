import matplotlib.pyplot as plt

files = ["thesis/scripts/cifar10_simple/test_wisard_test_10.txt", "thesis/scripts/cifar10_simple/test_wisard_test_15.txt",
    "thesis/scripts/cifar10_circular/test_wisard_test_10.txt", "thesis/scripts/cifar10_circular/test_wisard_test_15.txt"]

super_trt = []
super_trt_std = []
super_tt = []
super_tt_std = []
super_acc = []
super_acc_std = []

for i in range(len(files)):
    with open(files[i]) as f:
        trt = []
        trt_std = []
        tt = []
        tt_std = []
        acc = []
        acc_std = []

        for line in f:
            parameters = line.split(",")
            trt.append(round(float(parameters[1]), 2))
            trt_std.append(round(float(parameters[2]), 2))
            tt.append(round(float(parameters[3]), 2))
            tt_std.append(round(float(parameters[4]), 2))
            acc.append(round(float(parameters[5]), 2))
            acc_std.append(round(float(parameters[6]), 2))

        super_trt.append(trt)
        super_trt_std.append(trt_std)
        super_tt.append(tt)
        super_tt_std.append(tt_std)
        super_acc.append(acc)
        super_acc_std.append(acc_std)

plt.figure()
plt.errorbar(list(range(5,32)), super_trt[0], yerr=super_trt_std[0], label='simple 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_trt[1], yerr=super_trt_std[1], label='simple 15 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_trt[2], yerr=super_trt_std[2], label='circular 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_trt[3], yerr=super_trt_std[3], label='circular 15 bits', fmt='-o')
plt.legend()
plt.title('Training Time Comparison')
plt.xlabel('Address Size')
plt.ylabel('Training Time (s)')
plt.savefig("trt.png")

plt.figure()
plt.errorbar(list(range(5,32)), super_tt[0], yerr=super_tt_std[0], label='simple 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_tt[1], yerr=super_tt_std[1], label='simple 15 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_tt[2], yerr=super_tt_std[2], label='circular 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_tt[3], yerr=super_tt_std[3], label='circular 15 bits', fmt='-o')
plt.legend()
plt.title('Test Time Comparison')
plt.xlabel('Address Size')
plt.ylabel('Test Time (s)')
plt.savefig("tt.png")

plt.figure()
plt.errorbar(list(range(5,32)), super_acc[0], yerr=super_acc_std[0], label='simple 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_acc[1], yerr=super_acc_std[1], label='simple 15 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_acc[2], yerr=super_acc_std[2], label='circular 10 bits', fmt='-o')
plt.errorbar(list(range(5,32)), super_acc[3], yerr=super_acc_std[3], label='circular 15 bits', fmt='-o')
plt.legend()
plt.title('Accuracy Comparison')
plt.xlabel('Address Size')
plt.ylabel('Accuracy(%)')
plt.savefig("acc.png")