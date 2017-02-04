# KNN-OCR

This is the project file for the thesis "K-Nearest Neighbor Method for Optical Character Recognition".

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For this program to work, you need to have a machine running Ubuntu 14.04 LTS.

### Installing

The program needs Python 2.7+ and OpenCV 3.0 to work properly. Please refer to [this awesome guide](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/) on how to install those.

## How to prepare the data

To prepare the training data for the program, you can use cut_raw.py module and gen_data.py module.

### Cut the raw images into separate character images

Please note that the raw images need to contain English character images from a-z, A-Z and 0-9. This raw image file should be put in train_images/raw folder.

```
    1. Go to the cut_raw.py file and modify the DATA variable to the raw images that you want to cut.

    2. Run cut_raw.py programs.

    3. The program will show you the raw image with a green box surrounding a character.

    4. Use the keyboard to specify which character it is.

    5. Repeat step 3 and 4 to all the containing characters.

    6. After completion, a folder of the cut character images will be created in train_images/cut.
```


### Combine the character images into training data

```
    1. Go to gen_data.py file and modify DATA_NAME and DATA_FONTS variables.

    2. Modify the FEATURE_METHOD variable to an appropriate one (see below).

    3. Run gen_data.py program.

    4. After completion, a folder should be created in train_data/ folder. That folder would contain two files: features.txt and labels.txt.
```

The following table explains how to use the FEATURE_METHOD variable:

|FEATURE_METHOD|      Meaning      | Data points |
|--------------|-------------------|-------------|
| ocr_knn.Ocr.5 | 5x5 pixel image|25|
| ocr_knn.Ocr.10 |  10x10 pixel image|100|
| ocr_knn.Ocr.20 |20x20 pixel image|400|
| ocr_knn.Ocr.30 |  30x30 pixel image |900|
| ocr_knn.Ocr.3_5 | 3x5 pixel image |15|
| ocr_knn.Ocr.10_15|  10x15 pixel image |150|
| ocr_knn.Ocr.20_30 | 20x30 pixel image |600|

## How to run the program

All the main function of the program is written and run by recognize.py file. By changing the variables in recognize.py file, different results can be achieved.

### Variables meaning

|Variables|      Meaning      | Examples|
|--------------|-------------------|-------------|
| DATAS | The data to be used, which consists  of the data name and method.|[('11_fonts_gray_10', ocr_knn.Ocr.SIMPLE_10)]|
| TESTS |  A list of tests to be carried out. Each test consists of an image and an optional result file.|[('gill-sans.png', 'abcxyz.txt'), ('calibri.png', 'abcxyz.txt')]|
| KS |A list of K values.|[3, 4, 5]|
| WEIGHTS |A list of weighting distances. The supported values are 'distance' and 'uniform'.|['distance']|
|ALGORITHMS | A list of algorithms to be used. The supported values are 'brute', 'kd_tree' and 'ball_tree'. |['brute']|
|TEST|Whether you want to see the internal process or not. (Only working properly with normal usage)|False|

### Normal-usage

If you want to use the program as-it-is, modify TESTS, KS, WEIGHTS and ALGORITHMS variables to contain only one instance.

Optionally, you can raise the flag TESTING to see the internal program working.

Example:

```
TESTING = True
KS = [3]
WEIGHTS = ['distance']
ALGORITHMS = ['brute']
```

### Statistical-usage

Please note that you do not raise the flag TESTING in this mode, otherwise it would be really slow to achieve the results.

In this mode, you can use multiple values for TESTS, KS, WEIGHTS and ALGORITHMS. Example:

```
TESTING = False
KS = [3, 4, 5]
WEIGHTS = ['distance', 'uniform']
ALGORITHMS = ['brute', 'kd_tree']
```

After completion of running, there will be two files generated: output.txt and data.xlsx.

The output.txt is the resulting file of each test case of your running program. It is basically what you will see on the command prompt windows.

The data.xlsx is an Excel file which contains the resulting statistical number for each test case, including running time and accuracy as such.

## Questions?

If you have any questions, please email me at npvinhphat@gmail.com for further inquiries.