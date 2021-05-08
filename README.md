# vBF_GPU

The optimized GPU kernel for vector bilateral filtering on hyperspectral image 
The image width/heigth/spectrum is defined in vBF_GPH.h file
This code is used for test the dc image from 

http://cobweb.ecn.purdue.edu/~biehl/Hyperspectral_Project.zip

To test the kernel please download convet the file form above link, copy the dc.tif file 
to the current folder, and convert the hyperspectral image to a txt file using 
```
python convertTIF2TXT.py
```
You will find a dc.txt file in this folder Then run the program using 

```
./vBF_GPU.oug -i [inputfile] -o [outputfile] -w [windowSize]
```
in this test case, if the window size is 21 by 21, the arguments is:

```
./vBF_GPU.oug -i dc.txt -o dc_out.txt -w 21
```

You can also using other python script to conver the .mat file from other
hypterspectral dataset to the .txt file and conduct the experiment


