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
# ops =  np.load('D://Suite2Pprocessedfiles//Glaucus//2022-03-28data//suite2p//plane2//ops.npy', allow_pickle=True)
# ops = ops.item()
#ops =  np.load('C://Temporary_Suite2P_output//Eos//20220228//suite2p//combined//ops.npy', allow_pickle=True)
#ops = ops.item()
# iscell = np.load('D:/Suite2Pprocessedfiles/SS057/20220112/suite2p/combined//iscell.npy', allow_pickle=True)

#faster way to load
#from D drive
animal=  'Hedes'
date= '2022-07-05'
#note: if experiment type not known, put 'suite2p' instead
experiment= 'suite2p'
plane_number= '1'

filePathops='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//ops.npy'
filePathF='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//F.npy'
filePathFneu='D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//'+experiment+ '//plane'+plane_number+'//Fneu.npy'
ops= np.load(filePathops, allow_pickle= True)
ops = ops.item()
F1= np.load(filePathF, allow_pickle=True)

Fneu = np.load(filePathFneu, allow_pickle=True)
# subtracted= F1-0.7*Fneu
# plt.imshow(subtracted)

# fig, axs = plt.subplots(1, 2, figsize=(30, 9))
# axs[0].plot(F1[31])
# axs[1].plot(Fneu[31])

# choosing one ROI
n=32

#converting frames to seconds
fs=15

#ROI_256 = np.array(F1[0])
#ROI_512 = np.array(F2[21])


#plotting one cell/ a few cells and comparing two analyses
# plt.plot(F1[5], c="r")
# plt.plot (F2[21], c="b")
plt.plot(np.array(range(len(F1[n])))/fs,F1[n], c="blue")
plt.plot(np.array(range(len(Fneu[n])))/fs,Fneu[n], c="black")
plt.axvline(x=2691, c="red", linestyle="dashed", linewidth = 1)    

#plt.plot(np.array(range(len(F2[17])))/6,F2[17], c="blue")
#plt.plot(np.array(range(len(Fneu[17])))/6,Fneu[17], c="turquoise")

fig, axs = plt.subplots(1, sharex=True, sharey=True)
#choose ROI
n1=32
n2=118
n3= 323
n_str= str(n)
# axs[0].plot(np.array(range(len(F1[n1])))/fs, F1[n1], c="blue")
# axs[0].plot(np.array(range(len(F1[n1])))/fs, Fneu[n1], c="magenta")  
#axs[0].axvline(x=2691, c="red", linestyle="dashed", linewidth = 1)    
# axs[1].plot(np.array(range(len(F1[n])))/fs, F1[n2], c="green")
# axs[1].plot(np.array(range(len(F1[n])))/fs, Fneu[n2], c="magenta")
# axs[2].plot(np.array(range(len(F1[n])))/fs, F1[n3], c="orange")
# axs[2].plot(np.array(range(len(F1[n])))/fs, Fneu[n3], c="magenta")
plt.subplots_adjust(wspace=0.7, hspace=0.7)
plt.xlabel("Time(s)")
plt.ylabel("Raw Flurescence Intensity")
#yaxis.set_label_coords(-.1, .1)
n_str= str(n)

# filePathplot= 'D://Suite2Pprocessedfiles//'+animal+ '//'+date+ '//ROI'+n_str+'.png'
# plt.savefig(filePathplot)
# #plt.plot(np.array(range(len(F2[0])))/7.27,F2[0], c="b")

# x_256= np.array(range(len(F1[5])))/3
# sns.relplot(data=np.transpose(F1), kind="line", height=5,legend=None, aspect=2)

# sns.relplot(data=np.transpose(Fneu), kind="line", height=5,legend=None, aspect=2)

# norm_F= normalise(F1)

# #plot and save raw fluorescence trace, set how many cells to display
# n=10
# for i in range(1,n):
#     plt.plot(F1[i])
    


# # #plot deconvolution
# # # for i in range(1,10):
# # #    plt.plot(spks[i])

# # # plt.imshow(F)
# # # #plt.imshow(Fneu)
# # # #plt.plot(F)

# # #plt.imsave('D://Suite2Pprocessedfiles//SS057//20220112//plots//subtracted.png', subtracted)