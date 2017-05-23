 # Fetal Heart Tools

This repository contains all the code to run the experiments in my DPhil thesis "Automated Analysis of Fetal Cardiac Ultrasound Videos". It implements a tool for automatically estimating information
about each frame in ultrasound videos of the fetal heart, along with the surrounding
experimental workflow. The rest of this readme assumes some familiarity with the work (please refer to the references below).

This repo is currently under construction from existing code,
and a full README will be added soon.

There are two parts to this repository:

- The `cpp` folder contains C++ code for the image analysis pipeline.
- The `scripts` folder contains a set of supporting Python scripts that automate
the training and testing of large experiments and provide other parts of the
workflow such as analysis of results.

## C++ Code

###### Portability

I use exclusively GNU/Linux-based operating systems for my work and therefore the entire pipeline has been thoroughly tested on Ubuntu and Manjaro (Arch). However, I have been careful to use cross-platform libraries everywhere, along with a cross-platform build tool (CMake) to ensure that there is no reason that the C++ code won't compile and run on Windows and MacOS. That said, I have not been able to test this. If you are using this code on other platforms please let me know how it works out!

###### Overview of the Code

The C++ code is organised into the following directories:

- `exec` : This contains the top-level code for four executables. There are separate train and test programmes using rotation invariant features and rectangular features, giving `train_rotinv`, `test_rotinv`, `train_square`, and `test_square`.
- `features` : This contains code for extracting features from the images to use
in the random forests models.
- `forests` : This contains code for the random forest models used for analysing
the images, based on my [canopy](https://github.com/CPBridge/canopy) library.
- `filters` : This contains code for implementing the particle filters that link
estimates over time.
- `utilities` : This contains utility functions and definitions that are used
throughout the project.


###### Dependencies

The following third party open-source C++ libraries are necessary to compile and
run the code in this project. They are all common and available on most platforms
(GNU/Linux, MacOS, and Windows):

* A C++ compiler supporting the C++14 standard (requires a relatively modern version of your compiler).
* A C++ compiler supporting the [OpenMP](http://openmp.org/wp/) standard (includes most major compilers on major platforms including MSVC, g++ and clang).
* The [OpenCV](http://opencv.org) library for computer vision. Tested on version 3.1.0 but most fairly recent
versions should be compatible. If you are using GNU/Linux, there will probably
be a suitable packaged version in your distribution's repository.
* The [boost](http://www.boost.org) [special functions](http://www.boost.org/doc/libs/1_62_0/libs/math/doc/html/special.html), [program options](http://www.boost.org/doc/libs/1_64_0/doc/html/program_options.html), [filesystem](http://www.boost.org/doc/libs/1_62_0/libs/filesystem/doc/index.htm)
and [system](http://www.boost.org/doc/libs/1_62_0/libs/system/doc/index.html) libraries. Again there are likely to be a suitable packaged versions on your GNU/Linux distribution. Typically it is straightforward to install the entire set of boost libraries together.
- The [Eigen](http://eigen.tuxfamily.org) library for linear algebra. Again there are packaged versions for
most package managers.

For example, if you are using Ubuntu or an Ubuntu-based GNU/Linux distribution you can get all the necessary third-party dependencies as follows (tested on Ubuntu 16.04):

```bash
$ sudo apt install build-essential libopencv-dev libboost-dev libeigen3-dev

```

Furthermore, you will need a copy of the following code found in my other repositories:

- [CPBridge/RIFeatures](https://github.com/CPBridge/RIFeatures) : An OpenCV/C++ implementation of rotation invariant feature extraction.
- [CPBridge/monogenic](https://github.com/CPBridge/monogenic) : An OpenCV/C++ implementation of the monogenic signal.
- [CPBridge/RIFeatures](https://github.com/CPBridge/canopy) : A header-only C++ library for random forests.

You can straightforwardly clone these from Github with commands like the following on Unix-like systems:

```bash
$ cd /path/to/a/suitable/location
$ git clone https://github.com/CPBridge/RIFeatures.git
$ git clone https://github.com/CPBridge/monogenic.git
$ git clone https://github.com/CPBridge/canopy.git
```

You **do not** need to build or install the code in the above repositories - everything is handled
by the build process for this repository. You just need to download the source code.

###### Compiling

You should be able to use [CMake](https://cmake.org/) to configure your platform's
build process. Perform an out-of-source build and tell CMake to find the source in this repository's
`cpp` directory. You will also need to set some variables within CMake as follows:

- `RIFEATURES_DIR` : the location of the root of the RIFeatures repository on your system.
- `MONOGENIC_DIR` : the location of the root of the monogenic repository on your system.
- `CANOPY_DIR` : the location of the root of the canopy repository on your system.

You can either do this within your preferred CMake GUI (and then hit configure and generate) or pass them in on the command line when you invoke CMake, e.g. if you put all the repositories in your home directory `/home/fred`, then you could use

```bash
$ cmake -D CANOPY_DIR=/home/fred/canopy -D RIFEATURES_DIR=/home/fred/RIFeatures -D MONOGENIC_DIR=/home/fred/monogenic home/fred/fetal_heart_analysis/cpp/
```

Once CMake has generated the relevant set of build files, you will need to use
your platform's tool to complete the build process. E.g. on GNU/Linux with GNU make all you need is:

```bash
$ make
```

###### Quick Install on Ubuntu and Derivatives

The following script handles the entire install process including installing all dependencies, cloning all relevant repositories from Github, and compiling the C++ parts of the project. Make sure you are in the directory that you wish to contain the code **before** running it:

```bash
# Install build dependencies
sudo apt install build-essential git cmake

# Install third party C++ libraries
sudo apt install libopencv-dev libboost-dev libeigen3-dev

# Install python dependencies
sudo apt install python-numpy python-scipy python-matplotlib

# Clone all the necessary repos
git clone https://github.com/CPBridge/RIFeatures.git
git clone https://github.com/CPBridge/monogenic.git
git clone https://github.com/CPBridge/canopy.git
git clone https://github.com/CPBridge/fetal_heart_analysis.git

# Make a build directory
mkdir build
cd build

# Run the build process
cmake -D CANOPY_DIR=../canopy -D RIFEATURES_DIR=../RIFeatures -D MONOGENIC_DIR=../monogenic ../fetal_heart_analysis/cpp/
make
```


## Scripts

###### Installation and Dependencies

To run the scripts you will need a Python 2.7 interpreter (I've been using 2.7 but it may work on earlier versions). You will also need the following very common third-party packages, which are available in all good package management systems:

- [numpy](www.numpy.org) : Basic package for numeric/scientific computing
- [scipy](https://www.scipy.org/) : Further numeric and scientific libraries
- [matplotlib](http://matplotlib.org/) : Package for plotting
- OpenCV : you will need to ensure that your installation of OpenCV (see above) is set up to include the python bindings for your interpreter. Often this will have been handled automatically by however you installed OpenCV.

Additionally, you need to make sure that the interpreter can find the `scripts/utilities` directory, as some scripts depend on the modules in there. You can do this by, for example, adding that directory to your `PYTHONPATH` environment variable either temporarily or permanently.

###### Overview

The scripts are placed into several subdirectories according to the task they perform:

- `masks` : Tool for creating image ROI masks.
- `datasets` : Scripts for creating and managing dataset files, which are used to train random forest models.
- `filter_fit` : Scripts for fitting particle filter models to training data.
- `run` : Scripts for coordination of cross-validation experiments with multiple parameter sets.
- `analysis` : Scripts for performing analysis of results and making plots and charts.
- `utilities` : These are python modules that are not intended to be called directly, but are used by scripts in the other directories.

## Workflow Overview

This is an overview of the full experimental workflow. For more details, you will need to consult the relevant source files (most scripts and executables have help available if you pass the `-h` option). Where code examples are given, they pertain to GNU/Linux-like operating systems and may differ slightly on other platforms.

###### 1. Obtain Labelled Training Data

This stage is not handled by the tools in this repository. The framework expects a dataset of videos of the fetal heart which meet the following criteria:

- Be in the `.avi` format.
- Be placed in a single directory.
- Have names of the form `<subject-id>_<no>.avi` where `<subject-id>` is an arbitrary string that is the same for all videos captured from the same subject, and `<no>` is a positive integer number (of any length) that differentiates between the different videos for each subject. E.g. `subject001_3.avi` for the third video from subject `subject001` or `123456_10.avi` for the 10th video from subject `123456`. (There is no requirement that numbers are consecutive, just unique).
- Be accompanied by a `.png` image file that represents the ROI mask of the videos for a single subject. The ROI mask image should be non-zero within the ultrasound fan area (the area containing actual image information) and zero elsewhere (in blank areas and areas containing textual information etc that should be ignored by the image processing pipeline). The masks should be stored in a separate directory and have the format `<subject-id>_mask.png`. You can use the simple `maskcrop.py` tool in the `masks` directory to help you manually crop a polygonal mask from a video file.

```bash
$ ./maskcrop.py /path/to/videos/subject001_1.avi /path/to/masks/subject001_mask.png
```

- Be annotated to give a corresponding track file (`.tk` file) of a particular format. The tool for performing this annotation is in a [separate repository](https://github.com/CPBridge/heart_annotation_tool), whose documentation also gives further details about the required format. Furthermore, if you wish to use structure tracking there must additionally be a structure track file (`.stk`), using the structure annotation tool in the above repository. The `.tk` and `.stk` files should have the same name as the video file, but with the `.avi` extension removed and replaced with the relevant extension.

###### 2. Create Dataset Files

Dataset files are files defining a number of training examples to use when training a random forest model. Creating a dataset file involves finding all possible positive training patches from the track files, choosing a random subset of them, and choosing random negative/background examples. Scripts for performing these tasks are in the `datasets` directory.

To create a dataset file to train the random forests for 'global' variable estimates (heart view class, orientation, cardiac phase), you can use the `heartdatasets.py` script. You need to pass in the location of the directory containing the track (`.tk`) files (first positional argument), the number of view classes (`-v` option) present in the annotations (excluding the background class), the number of training examples you want for each of the positive classes (`-n` option), the names of any subjects you want to exclude from the training set (`-e` option), the name of the directory containing the mask file (`-m` option), and the name of the output file (`-o` option). For example,

```bash
$ ./heartdatasets.py /path/to/tracks/ -v 3 -n 5000 -m /path/to/masks/ -o /path/to/output/dataset/file
```

If you are running a leave-one-out cross validation experiment, you can use the `-c` option to automatically generate one dataset file for each fold (each fold excludes a single subject, according to the subject-id part of the video's file name, and the relevant dataset file's name indicates which subject was excluded by appending `_ex<subject-id>`). There are some other options here (jittering the training data for example), that you can look into (use `-h` for a list). One particularly important one is the `-f` option which allows you to specify that the dataset should not contain examples from the first few frames of a video (`-f 1` will omit the first frame and so on). If you wish to train models using motion features, you will need to use this as motion is not available in the first frame.

The file `substrdatasetsfromtracks.py` creates dataset files for structures. The usage is very similar to `heartdatasets.py`, but some additional information is needed. The directories containing both the `.stk` and `.tk` files must be passed as the first and second positional arguments respectively. The third positional argument is the fraction of the heart radius to use as the patch size for detecting structures. The fourth is the name of structures file that defines which structures are used. For example,

```bash
$ ./substrdatasetfromtracks.py /path/to/struct_tracks/ /path/to/tracks/ 0.5 /path/to/structure_list -n 5000 -m /path/to/masks/ -o /path/to/output/dataset/file
```

However, most of the time when you are using structures, you will want matched datasets to train matched models for the global variables and the structures. `heartandsubsdataset.py` wraps these two tasks into one for you, creating both a heart dataset and a structures dataset, where the latter has `_subs` appended to the name. Its usage is very similar to `substrdatasetsfromtracks.py` :

```bash
./heartandsubsdataset.py /path/to/struct_tracks/ /path/to/tracks/ 0.5 /path/to/structure_list -n 5000 -v 3 -m /path/to/masks/ -o /path/to/output/dataset/file
```

###### 3. Train Random Forest Models

###### 4. Create Filter Definition Files

###### 5. Test the Models

###### 6. Summarise results

###### 7. Make Plots

## Models

Types of forest model:

Name | Filename | Description
-|-|-|
classifier | `<base>.tr` | A random forest classifier trained to distinguish the different view classes and a background class.
jointOrientationRegressor | `<base>_ori.tr` | A random forest classifier in which each leaf node also contains a regressor for each class for predicting the orientation using equivariant features from the RIF feature extraction process.
circularRegressor | `<base>_phase<n>.tr` | A circular regression forest for predicting the cardiac phase of a heart of class `n`. There is one such model per view class.
jointOriPhaseRegressor | `<base>_phaseori<n>.tr` | A circular regression forest for predicting the cardiac phase of a heart of class `n` with an additional regressor in each leaf node for predicting the orientation using equivariant features from the RIF feature extraction process. There is one such model per view class.
classifier (structures) | `<base>_subs.tr` | A random forest classifier trained to distinguish the different structures from each other and from a background class.

Types of filter model:

Name | Description | Created With | Parameters File
---|---|---|
classOriFilter | Models how the view class, position and orientation variables behave. |`fit_class_ori_filter.py`| `class_ori_filter_params`
phaseFilter | Models how the cardiac phase and cardiac phase rate variables behave.| `fit_phase_filter.py` | `phase_filter_params`
PartitionedStructuresFilter | Models the structures' positions using a partitioned particle filter. |`fit_partitioned_structs_filter.py`| `structures_filter_params`
PCAStructuresFilter | Models the structures' positions using a principal component decomposition. |`fit_pca_structs_filter.py`| `structures_filter_params`

Required filter models (apply for both `test_rotinv` and `test_square`).

Model | 0 | 1 | 2 | 3 | 4 | 5
-----||||
classOriFilterg |:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:
phaseFilter|||:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:
PartitionedStructuresFilter|||||:white_check_mark:|
PCAStructuresFilter||||||:white_check_mark:

###### Rotation Invariant Features

This table shows the models produced by the `train_rotinv` executable when called with different combinations of the `-a` (orientation) and `-p` (cardiac phase) options:

Model | *none* | `-a` | `-p` | `-ap`
-----||||
classifier|:white_check_mark:||:white_check_mark:|:white_check_mark:
jointOrientationRegressor||:white_check_mark:||
circularRegressor|||:white_check_mark:|
jointOriPhaseRegressor||||:white_check_mark:

Forest models required during test time for different problem types (values of the `-p` option):

Model | 0 | 1 | 2 | 3 | 4 | 5
-----||||
classifier|:white_check_mark:|:white_check_mark:||:white_check_mark:|:white_check_mark:|:white_check_mark:
jointOrientationRegressor|||:white_check_mark:|||
circularRegressor||:white_check_mark:||||
jointOriPhaseRegressor||||:white_check_mark:|:white_check_mark:|:white_check_mark:
classifier (structures)|||||:white_check_mark:|:white_check_mark:


###### Rectangular Features

This table shows the models produced by the `train_square` executable when called with different combinations of the `-a` (orientation) and `-p` (cardiac phase) options:

Model | *none* | `-p`
-----||
classifier|:white_check_mark:|:white_check_mark:
circularRegressor||:white_check_mark:

Forest models required during test time for different problem types (values of the `-p` option):

Model | 0 | 1 | 2 | 3 | 4 | 5
-----||||
classifier|:x:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:
circularRegressor|:x:|:x:||:white_check_mark:|:white_check_mark:|:white_check_mark:
classifier (structures)|:x:|:x:|||:white_check_mark:|:white_check_mark:

The :x: symbol means that the problem type is not implemented with rectangular features.

## References

This code relates to the following publications:

- C.P. Bridge, C. Ioannou, and J.A. Noble, “Automated Annotation and Quantitative Description of Ultrasound Videos of the Fetal Heart”, *Medical Image Analysis 36* (February 2017) pp. 147-161. Open access available [here](http://dx.doi.org/10.1016/j.media.2016.11.006).
