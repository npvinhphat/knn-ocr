''' Utility helpers for OCR classification.
'''

import cv2
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import enum

# WHETHER WE PUT THE MODE TESTING ON
TESTING = True

# Image size to resized
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

""" Enums
"""
class Ocr(enum.Enum):
    PROJECTION = 1
    CONTOUR = 2

class Mode(enum.Enum):
    OPEN = 0
    CLOSE = 1

# Object
class Box(object):

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class TextBlock(object):
    """ A class for storing a block of text from an image
    """

    def __init__(self, img, box):
        """
        Initialize a TextBlock
        :param img: The image
        :param box: A Box class for dimensions (x, y, w, h)
        :param method: method to extract lines from the block
        """
        self.img = img
        self.box = box
        self.textLines = []

    def get_text_lines(self, method=Ocr.PROJECTION):
        """ Get all the text lines and store inside self.textLines
        """
        if len(self.textLines) != 0:
            raise ValueError('self.textLines already achieved!')

        line_boxes = []
        lines = []
        if method == Ocr.PROJECTION:
            line_boxes = self._get_boxes_by_projection(threshold=250)
        else:
            raise ValueError('Invalid method in get_text_lines: ' + str(method))

        for line_box in line_boxes:
            crop_img = self.img[line_box.y: line_box.y + line_box.h][line_box.x: line_box.x + line_box.w]
            lines.append(TextLine(crop_img, line_box))

        # Plot the process

        if TESTING:
            text_image_copy = self.img.copy()
            for l in line_boxes:
                cv2.rectangle(text_image_copy, (l.x, l.y), (l.x + l.w, l.y + l.h), (0, 255, 0), 1)
            cv2.imshow('find_characters', text_image_copy)
            cv2.waitKey(0)


        self.textLines = lines

    def _get_boxes_by_projection(self, threshold=250):
        # Reduce the gray image into horizontal projection
        reduced = cv2.reduce(self.img, 1, cv2.REDUCE_AVG)
        rows, cols = self.img.shape

        # Layout like a 1D image
        horizontal_projection = [x[0] for x in reduced]
        print type(horizontal_projection)

        lines = []
        last_mode = Mode.CLOSE
        line = Box()
        for i in range(len(horizontal_projection)):
            # If the pixels here is considerate
            if (horizontal_projection[i] < threshold):
                # if last mode is closing, create a character here
                if last_mode == Mode.CLOSE:
                    line = Box(x=0, y=i, w=cols, h=0)
                    last_mode = Mode.OPEN
            # If the pixels here is kinda blank
            else:
                # Register the current character here
                if last_mode == Mode.OPEN:
                    # This is the width
                    line.h = i - line.y
                    lines.append(line)
                    last_mode = Mode.CLOSE

        # Case when the mode is still open
        if last_mode == Mode.OPEN:
            line.w = len(horizontal_projection) - line.y
            lines.append(line)


        if TESTING:
            plt.figure()
            plot1 = plt.subplot('211')
            plt.plot(horizontal_projection)
            plt.subplot('212')
            plt.imshow(self.img, cmap='gray')
            plt.show()
            cv2.waitKey(0)
            plt.close()


        return lines


class TextLine(object):
    """ A class to store a block of line for an image.
    """

    def __init__(self, img, box):
        """
        Initialize and TextBlock
        :param img: The image
        :param box: A Box class for dimensions (x, y, w, h)
        :param method: method to extract lines from the block
        """
        self.img = img
        self.box = box
        self.textChars = []

    def get_text_chars(self, method=Ocr.PROJECTION):
        """ Return all the text chars and store inside self.textChars
        """
        if len(self.textChars) != 0:
            raise ValueError('self.textChars already achieved!')

        character_boxes = []
        characters = []

        if method == Ocr.PROJECTION:
            character_boxes = self._get_boxes_by_projection(threshold=250)
        elif method == Ocr.CONTOUR:
            character_boxes = self._get_boxes_by_contour()
        else:
            raise ValueError('Invalid method in find_characters: ' + str(method))

        # Plot the process
        '''
        if TESTING:
            for c in character_boxes:
                cv2.rectangle(line_image_copy, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), 1)
                cv2.imshow('find_characters', line_image_copy)
                cv2.waitKey(0)
        '''

        for character_box in character_boxes:
            crop_img = self.img[character_box.y: character_box.y + character_box.h,
                              character_box.x: character_box.x + character_box.w]
            characters.append(TextChar(crop_img, character_box))

        self.textChars = characters

    def _get_boxes_by_projection(self, threshold = 250):
        reduced = cv2.reduce(self.img, 0, cv2.REDUCE_AVG)
        rows, cols = self.img.shape

        vertical_projection = [x for x in reduced[0]]
        characters = []
        last_mode = Mode.CLOSE
        character = Box()
        for i in range(len(vertical_projection)):
            # If the pixels here is considerate
            if (vertical_projection[i] < threshold):
                # if last mode is closing, create a character here
                if last_mode == Mode.CLOSE:
                    character = Box(x=i, y=0, w=0, h=rows)
                    last_mode = Mode.OPEN
            # If the pixels here is kinda blank
            else:
                # Register the current character here
                if last_mode == Mode.OPEN:
                    # This is the width
                    character.w = i - character.x
                    characters.append(character)
                    last_mode = Mode.CLOSE

        # Case when the mode is still open
        if last_mode == Mode.OPEN:
            character.w = cols - character.y
            characters.append(character)

        '''
        if TESTING:
            plt.figure()
            plot1 = plt.subplot('211')
            plt.plot(vertical_projection)
            plt.subplot('212', sharex=plot1)
            plt.imshow(gray_image, cmap='gray')
            plt.show()
        '''

        return characters

    def _get_boxes_by_contour(self):
        characters = []
        blur_image = cv2.GaussianBlur(self.img, (3, 3), 0)
        thresh_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
                                             2)
        thresh_image_copy = thresh_image.copy()
        _, contours, _ = cv2.findContours(thresh_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            characters.append(Box(x, y, w, h))

        return characters


class TextChar(object):
    """A class to represent a character image.
    """

    def __init__(self, img, box):
        self.img = img
        self.box = box
        self.char = None

    def recognize_char(self, knn):
        self.char = knn.recognize(normalize_image(self.img))


class OcrKnn(object):
    """A class to represent the Knn of the system."""

    def __init__(self, classifications, flattened_images, k = 5):
        self.classifications = classifications
        self.flattened_images = flattened_images
        self.k = k
        self.knn = None

    def create_and_train(self):
        """ Create a Knn instance of opencv, then train the knn
        """
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.flattened_images, cv2.ml.ROW_SAMPLE, self.classifications)

    def recognize(self, image):
        """ Return a tuple of the recognize character.
        """
        res = np.float32(image.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)))
        ret, results, neighbors, dists = self.knn.findNearest(res, self.k)
        return str(chr(int(results[0][0])))


# Global method
def normalize_image(gray_img):
    """Global method to get a normalize image for data set. Use for character image only."""
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized_img = cv2.resize(binary_img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    return resized_img


def nothing(x):
    pass


# Main
def main():
    image = cv2.imread('tests/test_quote.jpg')
    if image is None:
        raise ValueError('image Not Found!')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    # Print all the line
    textBlock = TextBlock(gray, Box(0, 0, cols, rows))
    cv2.imshow('textBlock', textBlock.img)
    cv2.waitKey(0)
    textBlock.get_text_lines(method=Ocr.PROJECTION)
    # for textLine in textBlock.textLines:
        # cv2.imshow('textLine', textLine.img)
        # cv2.waitKey(0)

    # Print all the characters
    for textLine in textBlock.textLines:
        cv2.imshow('textLine', textLine.img)
        cv2.waitKey(0)
        textLine.get_text_chars(method=Ocr.CONTOUR)
        # for textChar in textLine.textChars:
            # cv2.imshow('textChar', textChar.img)
            # cv2.waitKey(0)

    # Recognize all the characters
    try:
        classifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return
    try:
        flattened_images = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return

    # Reshape numpy array to 1-d, necessary to pass to call to train
    classifications = classifications.reshape((classifications.size, 1))

    # Create OcrKnn instance
    ocrKnn = OcrKnn(classifications, flattened_images, k=5)
    ocrKnn.create_and_train()

    # Get the result
    for textLine in textBlock.textLines:
        cv2.imshow('textLine', textLine.img)
        # textLine.get_text_chars(method=Ocr.PROJECTION)
        for textChar in textLine.textChars:

            # Recognize the character
            textChar.recognize_char(ocrKnn)
            print textChar.char


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()