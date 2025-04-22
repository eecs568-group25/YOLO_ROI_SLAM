# YOLO_ROI_SLAM 

**This is an enhanced version of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) that incorporates a [YOLOv5](https://github.com/ultralytics/yolov5)-based object detection module. YOLOv5 enables real-time detection of common dynamic objects such as people, cars, bicycles, and animals. By selectively filtering out feature points associated with these detected classes, the SLAM system can effectively ignore transient elements that do not contribute to a stable map.**

Example results: 
<p align="center">
  <img src="example.gif" alt="YOLO_ROI_SLAM Demo" width="90%">
</p>



## Getting Started

* Operating System: Ubuntu 20.04
* Install ORB-SLAM3 prerequisites: C++11 or C++0x Compiler, Pangolin, OpenCV, Eigen3 and Python
* Install libtorch and move to Thirdparty folder
```bash
mv libtorch/ PATH/YOLO_ROI_SLAM/Thirdparty/ 
```

* Download and Build the project

```bash
cd YOLO_ROI_SLAM
chmod +x build.sh
./build.sh
```



## Download Dataset

### TUM

* Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
* Associate RGB images and depth images. Associations files are in the folder `./Examples/RGB-D/associations/` for the TUM dynamic sequences.
```bash
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```

### KITTI

* Download a sequence from https://www.cvlibs.net/datasets/kitti/eval_odometry.php. 
* Find groundtruth.txt for later use. 

### EUROC

* Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
* Find groundtruth.txt for later use. 




## Run YOLO_ROI_SLAM
Modes: RGBD, Stereo, Monocular

### TUM Dataset and RGB-D mode - example commands

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```

### KITTI dataset and stereo mode - example commands

```bash
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml PATH_TO_SEQUENCE_FOLDER 
```




## Do Evaluation
We use [evo](https://github.com/MichaelGrupp/evo3) to evaluate our results. It is a widely used trajectory evaluation toolkit for visual odometry and SLAM systems.

### Install evo

```bash
pip install evo
```

### Plot tum trajectories and the ground truth - example command

```bash
evo_traj tum --ref groundtruth.txt KeyFrameTrajectory.txt -as --plot
```

### Calculate the absolute pose error and save the results to .zip files - example command

```bash
mkdir results_tum
evo_ape tum groundtruth.txt KeyFrameTrajectory.txt --plot --align --plot_mode xz --save_results results_tum/TUM.zip
```

### Compare multiple results in the same results folder
If you also run the same tum dataset with the original ORB-SLAM3 following the steps above, get a TUM_slam3.zip and put into your YOLO_ROI_SLAM/results/ forder, you can compare the two results:

```bash
evo_res results/*.zip -p
```

