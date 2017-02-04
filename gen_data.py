""" Utility to generate data from cut images.
The input will be a list of data to be include inside the training sets.
For now, the input is only plain WIDTH * HEIGHT point
TODO : create more interesting point
The output will be two files: classifications.txt and flattened_images.txt store in train_data/<data_name>/
"""

import sys
import numpy as np
import cv2
import os
import ocr_knn
import glob

# Modify this to have different name for your data
DATA_NAME = '11_fonts_gray_5'

# Modify this to include different fonts
DATA_FONTS = ['gill-sans', 'calibri', 'arial', 'times-new-roman', 'rockwell', 'lucida-sans', 'adabi-mt', 'tw-cen-mt',
              'cambria', 'news-gothic-mt', 'candara']

# Modify this to use different type of features
# Two supported : 'simple', 'hog'
FEATURE_METHOD = ocr_knn.Ocr.SIMPLE_BIN_20_30

TRAIN_DATA_PATH = 'train_data'
CUT_IMAGE_PATH = 'train_images/cut'


def get_character(file_path):
    """Get the file name without the extension. This will return the first character in the file path.
    Ex: 'a_arial.png' => 'a'"""
    # Get the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name[0]

def main():
    # Initialize the storage
    features = None
    if FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_3_5 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_3_5:
        features = np.empty((0, 3 * 5))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_5 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_5:
        features = np.empty((0, 5 * 5))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_10 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_10:
        features = np.empty((0, 10 * 10))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_20 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_20:
        features = np.empty((0, 20 * 20))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_30 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_30:
        features = np.empty((0, 30 * 30))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_20_30 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_20_30:
        features = np.empty((0, 20 * 30))
    elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_10_15 or FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_10_15:
        features = np.empty((0, 10 * 15))
    elif FEATURE_METHOD == ocr_knn.Ocr.HOG:
        # SPLIT_N is the number of image split, each split image is computed by BIN_N bin of hog intensity
        features = np.empty((0, ocr_knn.SPLIT_N * ocr_knn.BIN_N))
    labels = []

    for data_font in DATA_FONTS:
        data_font_path = os.path.join(CUT_IMAGE_PATH, data_font)

        # Check if the font has been cut before hand
        if not os.path.isdir(data_font_path):
            raise ValueError('The font %s cannot be found!' % data_font)

        # Iterate all the character image inside the data_font_path
        image_paths = glob.glob(os.path.join(data_font_path, '*.png'))
        for image_path in image_paths:
            # Find the ord value of the character
            character = get_character(image_path)
            char_int = ord(character)

            # Find the flattened image
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = None
            if FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_3_5:
                feature = ocr_knn.preprocess_simple(gray_image, (3, 5))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_5:
                feature = ocr_knn.preprocess_simple(gray_image, (5, 5))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_10:
                feature = ocr_knn.preprocess_simple(gray_image, (10, 10))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_20:
                feature = ocr_knn.preprocess_simple(gray_image, (20, 20))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_30:
                feature = ocr_knn.preprocess_simple(gray_image, (30, 30))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_20_30:
                feature = ocr_knn.preprocess_simple(gray_image, (20, 30))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_10_15:
                feature = ocr_knn.preprocess_simple(gray_image, (10, 15))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_3_5:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (3, 5))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_5:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (5, 5))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_10:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (10, 10))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_20:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (20, 20))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_30:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (30, 30))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_20_30:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (20, 30))
            elif FEATURE_METHOD == ocr_knn.Ocr.SIMPLE_BIN_10_15:
                feature = ocr_knn.preprocess_simple_binary(gray_image, (10, 15))
            elif FEATURE_METHOD == ocr_knn.Ocr.HOG:
                feature = ocr_knn.preprocess_hog(gray_image)

            # Add inside the data list
            labels.append(char_int)
            features = np.append(features, feature, 0)


    # Convert to float32
    labels = np.array(labels, np.float32)
    # Flatten to 1d
    labels = labels.reshape((labels.size, 1))

    n, _ = labels.shape
    print 'Training complete with %d data points!!!' % n
    print 'Saving the result...'

    # Store the result
    new_path = os.path.join(TRAIN_DATA_PATH, DATA_NAME)
    if (os.path.isdir(new_path)):
        raise ValueError('The path %s has already exists! Please delete it before proceed!' % new_path)
    os.makedirs(new_path)
    np.savetxt(os.path.join(new_path, 'labels.txt'), labels)
    np.savetxt(os.path.join(new_path, 'features.txt'), features)

    print 'Result saved at %s!' % new_path

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




