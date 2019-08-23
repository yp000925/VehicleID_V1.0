import  numpy as np

file = np.genfromtxt('train_test_split/test_list_800.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_800.txt',query,fmt='%s',delimiter=' ')


file = np.genfromtxt('train_test_split/test_list_1600.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_1600.txt',query,fmt='%s',delimiter=' ')

file = np.genfromtxt('train_test_split/test_list_2400.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_2400.txt',query,fmt='%s',delimiter=' ')

file = np.genfromtxt('train_test_split/test_list_3200.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_3200.txt',query,fmt='%s',delimiter=' ')

file = np.genfromtxt('train_test_split/test_list_6000.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_6000.txt',query,fmt='%s',delimiter=' ')


file = np.genfromtxt('train_test_split/test_list_13164.txt',dtype='|U',delimiter=' ')
fids, pids =file.T
query_pids, idx = np.unique(pids,return_index=True)
query_fids = fids[idx]
query = np.array([query_fids,query_pids]).T
np.savetxt('train_test_split/query_list_13164.txt',query,fmt='%s',delimiter=' ')
