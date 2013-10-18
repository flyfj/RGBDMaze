This is matlab code to compute dense (per-pixel) labeling for color images as
well as RGB-D (Kinect) frames, which implements parts of the algorithms
detailed in our CVPR '12 paper:

    Xiaofeng Ren, Liefeng Bo, Dieter Fox
    RGB-(D) Scene Labeling: Features and Algorithms
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June, 2012.

The core features used in our scene labeling are the RGB-(D) kernel
descriptors. More information about kernel descriptors is available here:

http://www.cs.washington.edu/robotics/projects/kdes/

This software is developed by Xiaofeng Ren at Intel Labs and released under the
BSD license; see COPYRIGHT.txt. The current version is v1.0.1.

This demo code runs on two datasets: the Stanford Background dataset, for color
images of outdoor scenes; and the NYU Depth dataset (v1), for RGB-D frames of
indoor scenes. Scripts to download these datasets and convert into custom
formats, along with precomputed gpb-ucm segmentations, can be found as separate
files in the same directory as this code:

stanford.zip
nyu_depth.zip

Extract these files into the root directory, and run convert_dataset.m.

To run the demo code, go into ./code, and run

script_run_superpixel_labeling_stanford
script_run_superpixel_labeling_nyu_depth

In these scripts, you can choose to either run single-layer superpixel
classification or segmentation tree classification.  

Training uses a modified version of the liblinear software (included), which
takes weights for data points and uses dense matrices of float values.

In our CVPR paper, we also implemented a simple MRF model on top of the
superpixel classifications to further enforce smoothness. It is not included
here, but should be easy to add in, which provides another boost in accuracy.

Note that the kernel descriptors are not optimized for speed and are slow to
compute especially for the NYU Depth experiments (where we compute four types
of descriptors). It's recommended to compute the base kdes features in parallel
(with matlabpool). The NYU features take about a day when using 8 workers.

I will not be able to provide support for this code. Nonetheless, if you have
questions or suggestions, email me at xren@cs.washington.edu .

