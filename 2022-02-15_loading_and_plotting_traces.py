import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#use seaborn in the future!!

#normalizing the array
def normalise(data):
    return ((data - np.min(data))) / ((np.max(data) - np.min(data)))

#adjust file locations as needed, there will be the same set of data for every plane
#F = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//plane1//F.npy', allow_pickle=True)
#F_chan2 = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//F_chan2.npy', allow_pickle=True)
#Fneu = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//plane1//Fneu.npy', allow_pickle=True)
#spks = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//spks.npy', allow_pickle=True)
#stat = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//stat.npy', allow_pickle=True)
ops =  np.load('D://Suite2Pprocessedfiles//Eos//20220228//suite2p//combined//ops.npy', allow_pickle=True)
ops = ops.item()
ops =  np.load('C://Temporary_Suite2P_output//Eos//20220228//suite2p//combined//ops.npy', allow_pickle=True)
ops = ops.item()
# iscell = np.load('D:/Suite2Pprocessedfiles/SS057/20220112/suite2p/combined//iscell.npy', allow_pickle=True)


# subtracted= F-Fneu
# # plt.imshow(subtracted)

# fig, axs = plt.subplots(1, 2, figsize=(30, 9))
# axs[0].plot(F[31])
# axs[1].plot(Fneu[31])

# plt.plot(F[390])
# plt.plot(Fneu[390])

    


#sns.relplot(data=np.transpose(F), kind="line", height=5,legend=None, aspect=2)

#sns.relplot(data=np.transpose(Fneu), kind="line", height=5,legend=None, aspect=2)

# norm_F= normalise(F)

# #plot and save raw fluorescence trace, set how many cells to display
# n=10
# for i in range(1,n):
#     plt.plot(F[i])

#plt.savefig('D://Figures//SS057//ROI_390.png')

# #plot deconvolution
# # for i in range(1,10):
# #    plt.plot(spks[i])

# # plt.imshow(F)
# # #plt.imshow(Fneu)
# # #plt.plot(F)

# #plt.imsave('D://Suite2Pprocessedfiles//SS057//20220112//plots//subtracted.png', subtracted)