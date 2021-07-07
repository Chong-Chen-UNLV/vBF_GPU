# vBF_GPU

An optimized GPU kernel for vector bilateral filtering on hyperspectral image.
The details of the algorithm is described in:
@article{chen2021acceleration,
  title={Acceleration of vector bilateral filtering for hyperspectral imaging with GPU},
  author={Chen, Chong},
  journal={International Journal of Circuit Theory and Applications},
  volume={49},
  number={5},
  pages={1502--1514},
  year={2021},
  publisher={Wiley Online Library}
}


In the example, the image width/heigth/spectrum is defined in vBF_GPH.h file.

This code can be used for test the dc image from 

http://cobweb.ecn.purdue.edu/~biehl/Hyperspectral_Project.zip

To test the kernel please download the file from above link, copy the dc.tif file 
to the same folder of vBF_GPU executable file, and convert the hyperspectral image to a txt file using 
```
python convertTIF2TXT.py
```
You will find a dc.txt file in this folder. Then run the program using 

```
./vBF_GPU.out -i [inputfile] -o [outputfile] -w [windowSize]
```
in this test case, if the window size is 21 by 21, the arguments is:

```
./vBF_GPU.oug -i dc.txt -o dc_out.txt -w 21
```

The python script convertMat2TXT.py in this project is provided to convert the .mat file from other
hypterspectral dataset to the .txt file, this script can be used to verify our implemtation in other
datasets. 


