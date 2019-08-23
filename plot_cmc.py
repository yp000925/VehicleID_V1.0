import json
with open('./experiments/MLSM_resNet/800test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50_mslm = cmc

with open('./experiments/MLSM_resNet101/800test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm = cmc

with open('./experiments/MLSM_mob3/800test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob3_mslm = cmc

with open('./experiments/MLSM_mobNet/800test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob5_mslm = cmc

with open('./experiments/resnet/800test_21000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50 = cmc

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res50[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='Res50+BH')
plt.plot(x[0:rank],mob3_mslm[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='MobV1+MSLM_3')
plt.plot(x[0:rank],mob5_mslm[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='MobV1+MSLM_5')
plt.plot(x[0:rank],res50_mslm[0:rank],color='y',linestyle='-',marker='d',linewidth=1,label='Res50+MSLM')
plt.plot(x[0:rank],res101_mslm[0:rank],color='m',linestyle='-',marker='*',linewidth=1,label='Res101+MSLM')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation on VehicleID small test dataset')
plt.legend()
plt.grid()
plt.savefig('cmc_for_vehicleid_small.eps',format='eps')

plt.show()


with open('./experiments/MLSM_resNet/1600test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50_mslm = cmc

with open('./experiments/MLSM_resNet101/1600test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm = cmc

with open('./experiments/MLSM_mob3/1600test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob3_mslm = cmc

with open('./experiments/MLSM_mobNet/1600test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob5_mslm = cmc

with open('./experiments/resnet/1600test_21000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50 = cmc

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res50[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='Res50+BH')
plt.plot(x[0:rank],mob3_mslm[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='MobV1+MSLM_3')
plt.plot(x[0:rank],mob5_mslm[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='MobV1+MSLM_5')
plt.plot(x[0:rank],res50_mslm[0:rank],color='y',linestyle='-',marker='d',linewidth=1,label='Res50+MSLM')
plt.plot(x[0:rank],res101_mslm[0:rank],color='m',linestyle='-',marker='*',linewidth=1,label='Res101+MSLM')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation on VehicleID medium test dataset')
plt.legend()
plt.grid()
plt.savefig('cmc_for_vehicleid_medium.eps',format='eps')

plt.show()


with open('./experiments/MLSM_resNet/2400test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50_mslm = cmc

with open('./experiments/MLSM_resNet101/2400test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm = cmc

with open('./experiments/MLSM_mob3/2400test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob3_mslm = cmc

with open('./experiments/MLSM_mobNet/2400test_50000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
mob5_mslm = cmc

with open('./experiments/resnet/2400test_21000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res50 = cmc

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res50[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='Res50+BH')
plt.plot(x[0:rank],mob3_mslm[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='MobV1+MSLM_3')
plt.plot(x[0:rank],mob5_mslm[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='MobV1+MSLM_5')
plt.plot(x[0:rank],res50_mslm[0:rank],color='y',linestyle='-',marker='d',linewidth=1,label='Res50+MSLM')
plt.plot(x[0:rank],res101_mslm[0:rank],color='m',linestyle='-',marker='*',linewidth=1,label='Res101+MSLM')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation on VehicleID large test dataset')
plt.legend()
plt.grid()
plt.savefig('cmc_for_vehicleid_large.eps',format='eps')

plt.show()



with open('./experiments/MLSM_resNet101/800test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)
cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm_small = cmc

with open('./experiments/MLSM_resNet101/1600test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm_medium = cmc

with open('./experiments/MLSM_resNet101/2400test_70000_evaluation.json', 'r') as f:
    loaded_json = json.load(f)

cmc = loaded_json['CMC']
cmc = [x*100 for x in cmc]
res101_mslm_large = cmc

x=np.arange(0,len(cmc))
x= x+1
rank=30
plt.figure(1)
plt.plot(x[0:rank],res101_mslm_small[0:rank],color='r',linestyle='-',marker='^',linewidth=1,label='VehicleID_small')
plt.plot(x[0:rank],res101_mslm_medium[0:rank],color='b',linestyle='-',marker='o',linewidth=1,label='VehicleID_medium')
plt.plot(x[0:rank],res101_mslm_large[0:rank],color='g',linestyle='-',marker='>',linewidth=1,label='VehicleID_large')

plt.xlabel('Rank')
plt.ylabel('Matching rate(%)')
plt.title('CMC evaluation on VehicleID with different test dataset sizes')
plt.legend()
plt.grid()
plt.savefig('cmc_for_vehicleid_diff_size.eps',format='eps')

plt.show()
