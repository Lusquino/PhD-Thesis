from os import listdir
import matplotlib.pyplot as plt

files = {}

wtrtime = []
wetrtime = []
wtstime = []
wetstime = []
wacc = []
weacc = []
ctrtime = [[],[],[]]
cetrtime = [[],[],[]]
ctstime = [[],[],[]]
cetstime = [[],[],[]]
cacc = [[],[],[]]
ceacc = [[],[],[]]

folder = "results_movielens"

for fileName in listdir(folder):
	if fileName.endswith('.txt'):
		spl = fileName.split('_')
		if len(spl) == 3:
			ds = spl[0]
			pp = spl[1]
			md = spl[2].split('.')[0]
			if ds not in files.keys():
				files[ds] = {pp : [md]}
			else:
				if pp not in files[ds]:
					files[ds][pp] = [md]
				else:
					files[ds][pp].append(md)

			#print(fileName)
			with open(folder + "/" + fileName) as f:
				if md == 'bagging':
					superDict = {}
					for line in f.readlines():
						spl2 = line.strip().split(',')
						for i in range(len(spl2)):
							spl2[i] = spl2[i].strip()
						if spl2[2] not in superDict.keys():
							superDict[spl2[2]] = { spl2[1].strip() : { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}}
						else:
							if spl2[1] not in superDict[spl2[2]].keys():
								superDict[spl2[2]][spl2[1]] = { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}
							else:
								superDict[spl2[2]][spl2[1]][spl2[0]] = [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]

					superStr = f'\\begin{{table}}[]\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}\n\\hline\nPartition & wl & Models & Acc & Acc var & Training time & Test time '
					
					for pt in superDict.keys():
						c0 = 0
						superStr += f'\\\\ \\hline\n\multirow{{6}}{{*}}{{{pt}}} '
						for wl in superDict[pt].keys():
							c1 = 0
							if c0 == 0:
								superStr += f'& \multirow{{3}}{{*}}{{{wl}}} '
								c0 = 1
							else:
								superStr += f'\\\\ \\cline{{2-7}}\n& \multirow{{3}}{{*}}{{{wl}}} '
							for model in superDict[pt][wl].keys():
								v = superDict[pt][wl][model]
								if c1 == 0:
									superStr += f'& {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f}  '
									c1 = 1
								else:
									superStr += f'\\\\ \\cline{{3-7}}\n & & {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f} '

					superStr += f'\\\\ \\hline\n\\end{{tabular}}\n\\end{{table}}'
					print(superStr)
					superStr = ''

				if md == 'boost':
					superDict = {}
					for line in f.readlines():
						spl2 = line.strip().split(',')
						for i in range(len(spl2)):
							spl2[i] = spl2[i].strip()
						if spl2[1] not in superDict.keys():
							superDict[spl2[1]] = { spl2[0] : [spl2[12],spl2[13],spl2[14],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8],spl2[9],spl2[10],spl2[11]]}
						else:
							superDict[spl2[1]][spl2[0]] = [spl2[12],spl2[13],spl2[14],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8],spl2[9],spl2[10],spl2[11]]

					superStr = f'\\begin{{table}}[]\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}\n\\hline\nwl & Models & Acc & Acc var & Training time & Validation time & Test Time '
					
					for wl in superDict.keys():
						c1 = 0
						superStr += f'\\\\ \\hline\n\multirow{{3}}{{*}}{{{wl}}} '
						for model in superDict[wl].keys():
							v = superDict[wl][model]
							if c1 == 0:
								superStr += f'& {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2f} $\\pm$ {float(v[7]):.2f} & {float(v[9]):.2e} $\\pm$ {float(v[10]):.2f} '
								c1 = 1
							else:
								superStr += f'\\\\ \\cline{{2-7}}\n  & {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2f} $\\pm$ {float(v[7]):.2f} & {float(v[9]):.2e} $\\pm$ {float(v[10]):.2f}'

					superStr += f'\\\\ \\hline\n\\end{{tabular}}\n\\end{{table}}'
					print(superStr)
					superStr = ''

				if md == 'borda':
					superDict = {}
					for line in f.readlines():
						spl2 = line.strip().split(',')
						for i in range(len(spl2)):
							spl2[i] = spl2[i].strip()
						if spl2[2] not in superDict.keys():
							superDict[spl2[2]] = { spl2[1].strip() : { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}}
						else:
							if spl2[1] not in superDict[spl2[2]].keys():
								superDict[spl2[2]][spl2[1]] = { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}
							else:
								superDict[spl2[2]][spl2[1]][spl2[0]] = [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]

					superStr = f'\\begin{{table}}[]\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}\n\\hline\nPartition & wl & Policy & Acc & Acc var & Training time & Test time '
					
					for pt in superDict.keys():
						c0 = 0
						superStr += f'\\\\ \\hline\n\multirow{{6}}{{*}}{{{pt}}} '
						for wl in superDict[pt].keys():
							c1 = 0
							if c0 == 0:
								superStr += f'& \multirow{{3}}{{*}}{{{wl}}} '
								c0 = 1
							else:
								superStr += f'\\\\ \\cline{{2-7}}\n& \multirow{{3}}{{*}}{{{wl}}} '
							for model in superDict[pt][wl].keys():
								v = superDict[pt][wl][model]
								if c1 == 0:
									superStr += f'& {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f}  '
									c1 = 1
								else:
									superStr += f'\\\\ \\cline{{3-7}}\n & & {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f} '

					superStr += f'\\\\ \\hline\n\\end{{tabular}}\n\\end{{table}}'
					print(superStr)
					superStr = ''

				if md == 'voting':
					superDict = {}
					superKleber = {}
					for line in f.readlines():
						flag = True
						spl2 = line.strip().split(',')
						for i in range(len(spl2)):
							spl2[i] = spl2[i].strip()
							if spl2[i] == 'plurality1':
								spl2[i] = 'all candidates'
							if spl2[i] == 'plurality2':
								spl2[i] = 'only ties'
							if spl2[i] == 'plurality3':
								flag=False
							if spl2[i] == 'plurality4':
								spl2[i] = 'threshold'
						if not flag:
							if spl2[2] not in superKleber.keys():
								superKleber[spl2[2]] = [[spl2[1],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8],spl2[9],spl2[10],spl2[11]]]
							else:
								superKleber[spl2[2]].append([spl2[1],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8],spl2[9],spl2[10],spl2[11]])
						else:
							if spl2[2] not in superDict.keys():
								superDict[spl2[2]] = { spl2[1].strip() : { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}}
							else:
								if spl2[1] not in superDict[spl2[2]].keys():
									superDict[spl2[2]][spl2[1]] = { spl2[0] : [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]}
								else:
									superDict[spl2[2]][spl2[1]][spl2[0]] = [spl2[9],spl2[10],spl2[11],spl2[3],spl2[4],spl2[5],spl2[6],spl2[7],spl2[8]]

					superStr = f'\\begin{{table}}[]\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}\n\\hline\nPartition & wl & Policy & Acc & Acc var & Training time & Test time '
					
					for pt in superDict.keys():
						c0 = 0
						superStr += f'\\\\ \\hline\n\multirow{{6}}{{*}}{{{pt}}} '
						for wl in superDict[pt].keys():
							c1 = 0
							if c0 == 0:
								superStr += f'& \multirow{{3}}{{*}}{{{wl}}} '
								c0 = 1
							else:
								superStr += f'\\\\ \\cline{{2-7}}\n& \multirow{{3}}{{*}}{{{wl}}} '
							for model in superDict[pt][wl].keys():
								v = superDict[pt][wl][model]
								if c1 == 0:
									superStr += f'& {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f}  '
									c1 = 1
								else:
									superStr += f'\\\\ \\cline{{3-7}}\n & & {model} & {float(v[0]):.2f} $\\pm$ {float(v[1]):.2f} & {float(v[2]):.2f} & {float(v[3]):.2f} $\\pm$ {float(v[4]):.2f} & {float(v[6]):.2e} $\\pm$ {float(v[7]):.2f} '

					superStr += f'\\\\ \\hline\n\\end{{tabular}}\n\\end{{table}}'
					print(superStr)
					
					superStr = f'\\begin{{table}}[]\n\\begin{{tabular}}{{|c|c|c|c|c|c|}}\n\\hline\nPartition & wl & Acc & Acc var & Training time & Test time '

					for pt in superKleber.keys():
						c0 = 0
						superStr += f'\\\\ \\hline\n\multirow{{2}}{{*}}{{{pt}}} '
						for v in superKleber[pt]:
							if c0 == 0:
								superStr += f'& {v[0]} & {float(v[7]):.2f} $\\pm$ {float(v[8]):.2f} & {float(v[9]):.2f} & {float(v[1]):.2f} $\\pm$ {float(v[2]):.2f} & {float(v[4]):.2e} $\\pm$ {float(v[5]):.2f}  '
								c0 = 1
							else:
								superStr += f'\\\\ \\cline{{2-6}}\n& {v[0]} & {float(v[7]):.2f} $\\pm$ {float(v[8]):.2f} & {float(v[9]):.2f} & {float(v[1]):.2f} $\\pm$ {float(v[2]):.2f} & {float(v[4]):.2e} $\\pm$ {float(v[5]):.2f}  '
					superStr += f'\\\\ \\hline\n\\end{{tabular}}\n\\end{{table}}'
					print(superStr)

				if md == 'wisard':
					for line in f.readlines():
						spl2 = line.split(',')
						for i in range(len(spl2)):
							spl2[i]=spl2[i].strip()
						wtrtime.append( float(spl2[1]) )
						wetrtime.append( float(spl2[2]) )
						wtstime.append( float(spl2[4]) )
						wetstime.append( float(spl2[5]) )
						wacc.append( float(spl2[7]) )
						weacc.append( float(spl2[8]) )

				if md == 'clus':
					for line in f.readlines():

						spl2 = line.split(',')
						for i in range(len(spl2)):
							spl2[i]=spl2[i].strip()
						ctrtime[int(spl2[1])-3].append( float(spl2[2]) )
						cetrtime[int(spl2[1])-3].append( float(spl2[3]) )
						ctstime[int(spl2[1])-3].append( float(spl2[5]) )
						cetstime[int(spl2[1])-3].append( float(spl2[6]) )
						cacc[int(spl2[1])-3].append( float(spl2[8]) )
						ceacc[int(spl2[1])-3].append( float(spl2[9]) )

#

plt.figure()

plt.errorbar(list(range(5,32)), wtrtime, yerr=wetrtime, label='wisard', fmt='-o')
plt.errorbar(list(range(5,32)), ctrtime[0], yerr=cetrtime[0], label='clus-3', fmt='-o')
plt.errorbar(list(range(5,32)), ctrtime[1], yerr=cetrtime[1], label='clus-4', fmt='-o')
plt.errorbar(list(range(5,32)), ctrtime[2], yerr=cetrtime[2], label='clus-5', fmt='-o')

plt.legend()
plt.title('Training Time Comparison')
plt.xlabel('Address Size')
plt.ylabel('Training Time (s)')
plt.show()
plt.savefig("traintime.png")

#

f = plt.figure(figsize=(15,15))

ax1 = f.add_subplot(212)

ax1.errorbar(list(range(5,32)), wtstime, yerr=wetstime, label='wisard', fmt='-o')
ax1.errorbar(list(range(5,32)), ctstime[0], yerr=cetstime[0], label='clus-3', fmt='-o')
ax1.errorbar(list(range(5,32)), ctstime[1], yerr=cetstime[1], label='clus-4', fmt='-o')
ax1.errorbar(list(range(5,32)), ctstime[2], yerr=cetstime[2], label='clus-5', fmt='-o')

# plt.legend()
# plt.title('Test Time Comparison')
# plt.xlabel('Address Size')
# plt.ylabel('Test Time (s)')
# plt.show()

ax2 = f.add_subplot(221)

ax2.errorbar(list(range(5,32)), wtstime, yerr=wetstime, label='wisard', fmt='-o')

# plt.legend()
# plt.title('Test Time Comparison')
# plt.xlabel('Address Size')
# plt.ylabel('Test Time (s)')
# plt.show()

ax3 = f.add_subplot(222)

ax3.errorbar(list(range(5,32)), ctstime[0], yerr=cetstime[0], label='clus-3', fmt='-o')
ax3.errorbar(list(range(5,32)), ctstime[1], yerr=cetstime[1], label='clus-4', fmt='-o')
ax3.errorbar(list(range(5,32)), ctstime[2], yerr=cetstime[2], label='clus-5', fmt='-o')

plt.legend()
plt.title('Test Time Comparison')
plt.xlabel('Address Size')
plt.ylabel('Test Time (s)')
plt.show()
plt.savefig("testtime.png")


#

plt.figure()

plt.errorbar(list(range(5,32)), wacc, yerr=weacc, label='wisard', fmt='-o')
plt.errorbar(list(range(5,32)), cacc[0], yerr=ceacc[0], label='clus-3', fmt='-o')
plt.errorbar(list(range(5,32)), cacc[1], yerr=ceacc[1], label='clus-4', fmt='-o')
plt.errorbar(list(range(5,32)), cacc[2], yerr=ceacc[2], label='clus-5', fmt='-o')

plt.legend()
plt.title('Accuracy Comparison')
plt.xlabel('Address Size')
plt.ylabel('Accuracy(%)')
plt.show()
plt.savefig("acc.png")