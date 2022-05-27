import numpy as np

F = np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//F.npy', allow_pickle=True)
Fneu = np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//Fneu.npy', allow_pickle=True)
spks = np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//spks.npy', allow_pickle=True)
stat = np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//stat.npy', allow_pickle=True)
ops =  np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//ops.npy', allow_pickle=True)
ops = ops.item()
iscell = np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//combined//iscell.npy', allow_pickle=True)
Channel_2_plane1= np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//plane1//F_chan2.npy', allow_pickle=True)
Channel_2_plane2= np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//plane2//F_chan2.npy', allow_pickle=True)
Channel_2_plane3= np.load('D://Suite2Pprocessedfiles//trying_different_settings//suite2p//plane3//F_chan2.npy', allow_pickle=True)


print(F)
