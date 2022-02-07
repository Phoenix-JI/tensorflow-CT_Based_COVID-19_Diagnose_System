#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from scipy import ndimage


# In[30]:


def getScan(scanPath):
    
    data = sitk.ReadImage(scanPath)
    scan = sitk.GetArrayFromImage(data)
    
    return scan


# In[31]:


def normalizeScan(volume):
    
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume-min)/(max-min)
    volume = volume.astype("float32")
    
    return volume


# In[32]:


def resizeScan(scan,target_shape):
        
    current_width,current_height,current_depth = scan.shape
        
    target_width,target_height,target_depth = target_shape
    
    # Compute depth factor by 1/(D/N) 
    # Refer to the paper Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction
    
    width_factor = 1/(current_width/target_width)
    height_factor = 1/(current_height/target_height)
    depth_factor = 1/(current_depth/target_depth)
    
    # ndimage.zoom(): The array is zoomed using spline interpolation of the requested order.
    #scan = ndimage.rotate(scan, 90, reshape=False)
    
    scan = ndimage.zoom(scan,(width_factor,height_factor,depth_factor),order = 1)
    
    #print(width_factor,height_factor,depth_factor)
    
    return scan


# In[33]:


def preprocessScan(scanPath,target_shape):
    
    scan = getScan(scanPath)
    
    if(scan.shape[0] != scan.shape[1]):
        scan = np.swapaxes(scan,0,1)  #(64,128,128) => (128,128,64)
        scan = np.swapaxes(scan,1,2)
        
    scan = normalizeScan(scan)
    
    scan = resizeScan(scan,target_shape)
  
    
    return scan


# In[34]:


#scan_ = preprocessScan("/Users/phoenixji/Desktop/LungData/MosMedData/CT-1/study_0256.nii.gz",(128,128,64))


# In[35]:


#scan_ .shape


# In[36]:


#plt.imshow(scan_[:,:,30])


# In[ ]:




