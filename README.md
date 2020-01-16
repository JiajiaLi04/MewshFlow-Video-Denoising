# MeshFlow Video Denoising
===============
# Image and Video Processing Lab

- Built by Jiajia Li from University of Electrical Science and Technology of China (UESTC), now a graduate student at Michigan State University.

Overview
-------
This is the source Code for MeshFlow Video Denoising ([PDF](https://ieeexplore.ieee.org/document/8296826)).
We Proposed an efficient video denoising approach by utilizing the MESHFLOW motion model for the camera motion compensation
The meshflow is a sparse motion field. It is used to estimate motions between neighboring frames, which are used to align frames within a sliding time window. This method could achieve strong denoising results under a very fast speed and is robust to different types of camera motions and scene types.

Project website can be found at [here](http://www.liushuaicheng.org/ICIP/2017/index.html).

Usage
-----
1. Create a new folder `build`<br>
2. Inside the folder, using the code below for cmake to build the project files:<br>
```bash
cmake DCMAKE_BUILD_TYPE=Release ..
```
3. Move the test video into this new folder and run this project.<br><br>

For `Visual Studio` users, when run the project file, you should change the `single startup project` option in the solution property into the correct one (not the `ALLBUILD.EXE`).<br>

For Windows users, using `Cmakelists.txt` as the cmakelist. (Remember to change the path for the OpenCV build folder.)<br>
For Linux users, using `Cmakelists_Linux.txt` as the cmakelist.<br><br>

## Citation
If you find this useful in your research, please cite our paper "Meshflow Video Denoising" ([PDF](https://ieeexplore.ieee.org/document/8296826):
~~~
@inproceedings{ren2017meshflow,
  title={Meshflow video denoising},
  author={Ren, Zhihang and Li, Jiajia and Liu, Shuaicheng and Zeng, Bing},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  pages={2966--2970},
  year={2017},
  organization={IEEE}
}
~~~
