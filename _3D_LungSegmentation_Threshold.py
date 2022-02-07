#!/usr/bin/env python
# coding: utf-8

# In[15]:


from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import scipy.ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def get_segmented_lungs(im, threshold=-400):
    

    binary = im < threshold


    cleared = clear_border(binary)


    label_image = label(cleared)
    

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0


    selem = disk(2)
    binary = binary_erosion(binary, selem)


    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary


# In[ ]:




