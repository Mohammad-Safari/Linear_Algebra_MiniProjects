# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SVD Decomposition Module
# %% [markdown]
# ## Denoising Noisy Image Frame from File

# %%
from matplotlib import pyplot as plt
img = plt.imread('noisy.jpg') # read picture from path
plt.imshow(img)
plt.show()

# %% [markdown]
# ### calculating matrix decomposition

# %%
import numpy as np
from numpy import linalg as la
from numpy import matrix as mt
dcm = {
    'U':[0, 0, 0],
    'S':[0, 0, 0],
    'V':[0, 0, 0],
    'R':[0, 0, 0]
}
for i in range(3):
    dcm['U'][i], dcm['S'][i], dcm['V'][i] = la.svd(img[:, :, i])

# %% [markdown]
# #### choosing a cuttoff value according to histogram

# %%
plt.hist(dcm['S'],range(0,10000,100))
plt.show()


# %%
plt.scatter(range(142), dcm['S'][0], color='red', s=1)
plt.scatter(range(142), dcm['S'][1], color='blue', s=1)
plt.scatter(range(142), dcm['S'][2], color='green', s=1)
plt.show()

# %% [markdown]
# ### Denoising Picture and Recostructing Image by New Sigma matrix

# %%
cutoff = [1300,1300,1300]
##
diff = img.shape[0]-img.shape[1]
diff = 0 if(diff < 0) else diff
##
for k in range(3):
    dcm['R'][k] = np.zeros(img.shape[1])
    for i in range(len(dcm['S'][k])):
        dcm['R'][k][i] = dcm['S'][k][i] if(dcm['S'][k][i] > cutoff[k]) else 0
    dcm['R'][k] = np.reshape(np.append(np.diag(dcm['R'][k]), np.zeros((diff, img.shape[1]))),(img.shape[0], img.shape[1]))


# %%
new_img = np.full((356,142,3),255);
for k in range(3):
    new_img[:,:,k] = np.matrix(dcm['U'][k])*np.matrix(dcm['R'][k])*np.matrix(dcm['V'][k])


# %%
plt.imshow(new_img)
plt.show()

# %% [markdown]
# ## Surface Denoise

# %%
import utility as ut
ut.show_main()


# %%
ut.show_noisy()


# %%
records = ut.get_function()
records.shape
fdc = {
    'U':0,
    'S':0,
    'V':0,
    'R':0
}
fdc['U'], fdc['S'], fdc['V'] = la.svd(records)

# %% [markdown]
# #### choosing a cuttoff value according to scatter

# %%
plt.scatter(range(30), fdc['S'], color='red', s=1)
plt.scatter(range(30), fdc['S'], color='blue', s=1)
plt.show()

# %% [markdown]
# ### Denoising Coordination and Reconstructing Function Coordinations

# %%
fcutoff = 1
##
diff = records.shape[0] - records.shape[1]
diff = 0 if(diff < 0) else diff
##
fdc['R'] = np.zeros(records.shape[1])
for i in range(len(fdc['S'])):
    fdc['R'][i] = fdc['S'][i] if(fdc['S'][i] > fcutoff) else 0
fdc['R'] = np.reshape(np.append(np.diag(fdc['R']), np.zeros((diff, records.shape[1]))),(records.shape[0], records.shape[1]))


# %%
new_func = np.matrix(fdc['U'])*np.matrix(fdc['R'])*np.matrix(fdc['V'])

# %% [markdown]
# #### comparing results

# %%
print('Noisy function:\n')
ut.show_noisy()
print('Denoised function:\n')
ut.show_my_matrix(new_func)


# %%



