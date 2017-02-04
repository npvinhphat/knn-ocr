"""
Utility to cut a single image to character image.
If the image is named "x.type" and the recognized character is "c"
then the resulting cut image will be "c.type" and stored in folder x
"""

import cv2
import ocr_knn
import os
import warnings
import sys
import shutil

PATH_TO_RAW = 'train_images/raw'
PATH_TO_CUT = 'train_images/cut'

# Modify this list to add your own data to cut
'''DATA = [x for x, y in [('gill-sans.png', 'abcxyz.txt'),
                 ('calibri.png', 'abcxyz.txt'),
                 ('arial.png', 'abcxyz.txt'),
                 ('times-new-roman.png', 'abcxyz.txt'),
                 ('rockwell.png', 'abcxyz.txt'),
                 ('lucida-sans.png', 'abcxyz.txt'),
                 ('adabi-mt.png', 'abcxyz.txt'),
                 ('tw-cen-mt.png','abcxyz.txt'),
                 ('cambria.png','abcxyz.txt'),
                 ('news-gothic-mt.png', 'abcxyz.txt'),
                 ('bodoni-72.png', 'abcxyz.txt'),
                 ('candara.png','abcxyz.txt')]]'''
DATA = ['arial.png']

# List of valid characters
VALID_CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def cut(raw_image):
    """
    Cut a raw_image into multiple character images.
    Return a tuple of cut_images and cut_labels.
    """
    cut_images = []
    cut_labels = []

    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray_image.shape

    # Print all the line
    textBlock = ocr_knn.TextBlock(gray_image, ocr_knn.Box(0, 0, cols, rows))
    # cv2.imshow('textBlock', textBlock.img)
    textBlock.get_text_lines(method=ocr_knn.Ocr.PROJECTION, params={'threshold':255})
    # for textLine in textBlock.textLines:
    # cv2.imshow('textLine', textLine.img)
    # cv2.waitKey(0)

    # Print all the characters
    for textLine in textBlock.textLines:
        # cv2.imshow('textLine', textLine.img)
        textLine.get_text_chars(method=ocr_knn.Ocr.COMBINE)
        textLineImg = textLine.img.copy()
        textLineImg = cv2.cvtColor(textLineImg, cv2.COLOR_GRAY2BGR)
        for textChar in textLine.textChars:
            cv2.imshow('textChar', textChar.img)
            cv2.rectangle(textLineImg, (textChar.box.x, textChar.box.y), (textChar.box.x + textChar.box.w,
                                                                          textChar.box.y + textChar.box.h), (0, 255, 0),
                          2)
            cv2.imshow('textLine', textLineImg)
            # Wait for input to be recognized
            char_int = cv2.waitKey(0) & 255
            while (True):
                # if Esc key
                if char_int == 27:
                    sys.exit()
                elif chr(char_int) in VALID_CHARS:
                    # Put the result
                    cut_images.append(textChar.img)
                    cut_labels.append(chr(char_int))
                    # Break out of the loop
                    break
                elif chr(char_int) == '/':
                    print 'Do not count this letter!'
                    break
                else:
                    print 'Invalid character input: ' + str(char_int)
                    print 'Try again!'
                char_int = cv2.waitKey(0) & 255

    # Return the result
    return (cut_images, cut_labels)


def main():
    # Get all the path from DATA
    for raw_data_name in DATA:
        raw_data_path = os.path.join(PATH_TO_RAW, raw_data_name)
        print 'Start to extract data from %s...' %  raw_data_path

        # Retrieve the images from raw_data_path
        image = cv2.imread(raw_data_path)
        if image is None:
            raise ValueError('Image of this path does not exist: %s' % raw_data_name)

        # Create a directory with the data name
        new_folder_name = os.path.splitext(raw_data_name)[0]
        new_path = os.path.join(PATH_TO_CUT, new_folder_name)
        if os.path.exists(new_path):
            raise ValueError('The path %s already exists. Please delete it first then try again.' % new_path)
        os.makedirs(new_path)

        # Retrieve all the cut image from the raw_image
        cut_images, cut_labels = cut(image)

        # For each image and label, create a file in new_path
        for i in range(len(cut_images)):
            cut_image_name = str(cut_labels[i]) + '_' + new_folder_name + '.png'
            cut_image_path = os.path.join(new_path, cut_image_name)
            cv2.imwrite(cut_image_path, cut_images[i])

        print 'Extract data from %s completed! The file now is stored in folder %s.' % (raw_data_path, new_path)

if __name__ == '__main__':
    main()