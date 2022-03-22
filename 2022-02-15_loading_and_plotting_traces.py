import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#use seaborn in the future!!

#normalizing the array
def normalise(data):
    return ((data - np.min(data))) / ((np.max(data) - np.min(data)))

#adjust file locations as needed, there will be the same set of data for every plane
# F1 = np.load('D://Suite2Pprocessedfiles//Eos//2022-03-04//te2p//plane0//F.npy', allow_pickle=True)
# F1= np.load('D://Suite2Pprocessedfiles//Eos//2022-03-09//suite2p//plane1//F.npy', allow_pickle=True)
# F2= np.load('D://Suite2Pprocessedfiles//Eos//2022-03-04//suite2p//plane1//F.npy', allow_pickle=True)
#F_chan2 = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//F_chan2.npy', allow_pickle=True)
# Fneu = np.load('D://Suite2Pprocessedfiles//Eos//2022-03-09//suite2p//plane1//Fneu.npy', allow_pickle=True)
# Fneu2= np.load('D://Suite2Pprocessedfiles//Eos//2022-03-04//suite2p//plane1//Fneu.npy', allow_pickle=True)
# spks = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//spks.npy', allow_pickle=True)
#stat = np.load('D://Suite2Pprocessedfiles//SS057//20220112//suite2p//combined//stat.npy', allow_pickle=True)
ops =  np.load('D://Suite2Pprocessedfiles//Hedes//2022-03-17//2022-03-17data//suite2p//plane1//ops.npy', allow_pickle=True)
ops = ops.item()
#ops =  np.load('C://Temporary_Suite2P_output//Eos//20220228//suite2p//combined//ops.npy', allow_pickle=True)
#ops = ops.item()
# iscell = np.load('D:/Suite2Pprocessedfiles/SS057/20220112/suite2p/combined//iscell.npy', allow_pickle=True)


# subtracted= F-Fneu
# # plt.imshow(subtracted)

# fig, axs = plt.subplots(1, 2, figsize=(30, 9))
# axs[0].plot(F[31])
# axs[1].plot(Fneu[31])

# # choosing one ROI
# n=2
# one_ROI= (F1[n])
# #converting frames to seconds
# fs=6

ROI_256 = np.array(F1[0])
#ROI_512 = np.array(F2[21])
n=8

#plotting one cell/ a few cells and comparing two analyses
# plt.plot(F1[5], c="r")
# plt.plot (F2[21], c="b")
plt.plot(np.array(range(len(F1[n])))/7.27,F1[n], c="r")
#plt.plot(np.array(range(len(Fneu[n])))/7.27,Fneu[n], c="magenta")
plt.plot(np.array(range(len(F2[17])))/6,F2[17], c="blue")
#plt.plot(np.array(range(len(Fneu[17])))/6,Fneu[17], c="turquoise")



plt.xlabel("Time(s)")
plt.ylabel("Flurescence Intensity")
#plt.plot(np.array(range(len(F2[0])))/7.27,F2[0], c="b")

# x_256= np.array(range(len(F1[5])))/3
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