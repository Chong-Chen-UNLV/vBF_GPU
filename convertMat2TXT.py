import numpy as np
import h5py
f = h5py.File('0001.mat','r')
data = f.get('hypercube')
data = np.array(data) # For converting to a NumPy array

f2 = open('0001.txt','w')
f2.write(str(data.shape[1]) + '\t' + str(data.shape[2]) + '\t' + str(data.shape[0]) + '\n')
for id in range(len(data)):
    if (id == 0):
        img = data[id,:,:]
    else:
        img = np.append(img, data[id,:,:], axis = 0)

np.savetxt(f2, img, fmt="%i")

f2.close()