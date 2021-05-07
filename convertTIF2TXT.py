import numpy as np
import struct
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo.gdalconst import *
import matplotlib.pyplot as plt
cube = gdal.Open('dc.tif')
f = open('dc.txt','w')
 
f.write(str(cube.RasterXSize) + '\t' + str(cube.RasterYSize) + '\t' + str(cube.RasterCount) + '\n')
for x in range(cube.RasterCount):
    bnd = cube.GetRasterBand(x+1)
    img = bnd.ReadAsArray(0,0,cube.RasterXSize, cube.RasterYSize)
    if(x == 0):
        imgAll = img
    else:
        imgAll = np.append(imgAll, img, axis = 0)

np.savetxt(f, imgAll, fmt="%i")
