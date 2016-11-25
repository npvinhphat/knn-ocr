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
DATA_NAME = 'ten_fonts'

# Modify this to include different fonts
DATA_FONTS = ['arial', 'calibri', 'cambria', 'century', 'comic-sans', 'gill-sans', 'helvetica', 'rockwell',
              'segoe-print', 'times-new-roman']

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
    flattened_images = np.empty((0, ocr_knn.RESIZED_IMAGE_WIDTH * ocr_knn.RESIZED_IMAGE_HEIGHT))
    classifications = []

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
            binary_image = ocr_knn.normalize_image(gray_image)
            flattened_image = binary_image.reshape((1, ocr_knn.RESIZED_IMAGE_HEIGHT * ocr_knn.RESIZED_IMAGE_WIDTH))

            # Add inside the data list
            classifications.append(char_int)
            flattened_images = np.append(flattened_images, flattened_image, 0)


    # Convert to float32
    classifications = np.array(classifications, np.float32)
    # Flatten to 1d
    classifications = classifications.reshape((classifications.size, 1))

    n, _ = classifications.shape
    print 'Training complete with %d data points!!!' % n
    print 'Saving the result...'

    # Store the result
    new_path = os.path.join(TRAIN_DATA_PATH, DATA_NAME)
    if (os.path.isdir(new_path)):
        raise ValueError('The path %s has already exists! Please delete it before proceed!' % new_path)
    os.makedirs(new_path)
    np.savetxt(os.path.join(new_path, 'classifications.txt'), classifications)
    np.savetxt(os.path.join(new_path, 'flattened_images.txt'), flattened_images)

    print 'Result saved at %s!' % new_path

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




