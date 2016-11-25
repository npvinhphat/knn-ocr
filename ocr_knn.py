""" Utility helpers for OCR classification.
"""

import cv2
import sys
import os
import numpy as np
import matplotlib
import operator
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import enum
import difflib
from sklearn import neighbors
from time import time
import warnings

# WHETHER WE PUT THE MODE TESTING ON
TESTING = True

# Image size to resized
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Lower percentage of space w.r.t. medium width
DEFAULT_SPACE_PERCENTAGE = 0.5

# Enums
class Ocr(enum.Enum):
    PROJECTION = 1
    CONTOUR = 2
    COMBINE = 3

    FULL = 1

    # Text Document method
    DILATION = 1


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

    def isAligned(self, other, horizontal = False):
        if horizontal:
            return not max(self.x, other.x) < min(self.x + self.w, other.x + other.w)
        else:
            return not max(self.y, other.y) < min(self.y + self.h, other.y + other.h)

    def combine(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return Box(x, y, w, h)


class TextDocument(object):
    """ A class for storing a single document file with several text blocks."""

    def __init__(self, img):
        """Initialize a text document."""
        self.img = img
        self.textBlocks = []

    def get_text_blocks(self, method=Ocr.DILATION, params=None):
        """ Get all the text blocks and store inside self.textBlocks."""
        if len(self.textBlocks) != 0:
            raise ValueError('self.textLines already achieved!')

        block_boxes = []
        blocks = []
        if method == Ocr.DILATION:
            block_boxes = self._get_text_block_by_dilation(params)
        else:
            raise ValueError('Invalid method in get_text_blocks: ' + str(method))

        for block_box in block_boxes:
            crop_img = self.img[block_box.y: block_box.y + block_box.h, block_box.x: block_box.x + block_box.w]
            blocks.append(TextBlock(crop_img, block_box))

        # Assign text block inside:
        self.textBlocks = blocks

    def _get_text_block_by_dilation(self, params=None):
        # Blur the image
        blur_img = cv2.GaussianBlur(self.img, (5, 5,), 0)
        _, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Default max components
        max_components = 4
        if params and 'max_components' in params:
            max_components = params['max_components']

        contours = self._find_components(thresh_img, max_components=max_components)
        block_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            block_boxes.append(Box(x, y, w, h))

        return block_boxes

    def _dilate(self, input, size, iterations=5):
        # Use a dilation in horizontal for bleeding technique
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
        dilated_image = cv2.dilate(input, morph_kernel, iterations=iterations)
        return dilated_image

    def _find_components(self, input_img, max_components=4):
        """Dilate an image until only max_components left."""
        count = sys.maxint
        iterations = 1
        size = (3, 5)
        contours = []
        # inverse input
        input_inverse = 255 - input_img
        while count > max_components:
            dilated_image = self._dilate(input_inverse, size, iterations=iterations)
            # inverse the dilated image, since find contours only find black pixel
            if TESTING:
                cv2.imshow('dilated_image', dilated_image)
                cv2.waitKey(0)
            _, contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
            iterations += 1
        return contours

    def get_result(self):
        # Sort text blocks first, in y direction
        self.textBlocks.sort(key= lambda x:x.box.y)
        result = ''
        for textBlock in self.textBlocks:
            result += textBlock.get_result()
        return result

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

    def get_text_lines(self, method=Ocr.PROJECTION, params=None):
        """ Get all the text lines and store inside self.textLines
        """
        if len(self.textLines) != 0:
            raise ValueError('self.textLines already achieved!')

        line_boxes = []
        lines = []
        if method == Ocr.PROJECTION:
            line_boxes = self._get_boxes_by_projection(params)
        else:
            raise ValueError('Invalid method in get_text_lines: ' + str(method))

        for line_box in line_boxes:
            crop_img = self.img[line_box.y: line_box.y + line_box.h, line_box.x: line_box.x + line_box.w]
            lines.append(TextLine(crop_img, line_box))

        # Plot the process

        '''
        if TESTING:
            text_image_copy = self.img.copy()
            for l in line_boxes:
                cv2.rectangle(text_image_copy, (l.x, l.y), (l.x + l.w, l.y + l.h), (0, 255, 0), 1)
            cv2.imshow('find_characters', text_image_copy)
            cv2.waitKey(0)
        '''

        self.textLines = lines

    def _get_boxes_by_projection(self, params):
        if (params and 'threshold' in params):
            threshold = params['threshold']
        else:
            threshold = 250

        # Reduce the gray image into horizontal projection
        reduced = cv2.reduce(self.img, 1, cv2.REDUCE_AVG)
        rows, cols = self.img.shape

        # Layout like a 1D image
        horizontal_projection = [x[0] for x in reduced]

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

        '''
        if TESTING:
            plt.figure()
            plot1 = plt.subplot('211')
            plt.plot(horizontal_projection)
            plt.subplot('212')
            plt.imshow(self.img, cmap='gray')
            plt.show()
            cv2.waitKey(0)
            plt.close()
        '''


        return lines

    def get_result(self):
        # Sort text blocks first, in y direction
        self.textLines.sort(key= lambda x:x.box.y)
        result = ''
        for textLine in self.textLines:
            result += textLine.get_result()
        result += '\n'
        return result


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

    def get_text_chars(self, method=Ocr.PROJECTION, params=None):
        """ Return all the text chars and store inside self.textChars
        """
        if len(self.textChars) != 0:
            raise ValueError('self.textChars already achieved!')

        character_boxes = []
        characters = []

        if method == Ocr.PROJECTION:
            character_boxes = self._get_boxes_by_projection(params)
        elif method == Ocr.CONTOUR:
            character_boxes = self._get_boxes_by_contour()
        elif method == Ocr.COMBINE:
            character_boxes = self._get_boxes_by_combine()
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

    def _get_boxes_by_projection(self, params):
        if params and 'threshold' in params:
            threshold = params['threshold']
        else:
            threshold = 250

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

    def _get_boxes_by_combine(self):
        # Get the boxes by contour first
        boxes = self._get_boxes_by_contour()
        # Sort the boxes by x
        sorted_boxes = sorted(boxes, key=operator.attrgetter('x'))
        boxes = []
        # Iterate the whole box to get combinable boxes
        for b in sorted_boxes:
            if len(boxes) != 0 and b.isAligned(boxes[-1], horizontal=False):
                boxes[-1] = boxes[-1].combine(b)
            else:
                boxes.append(b)
        # return result
        return boxes

    def _get_median_width(self):
        # Sort by width, then get the medium component
        temp = sorted(self.textChars, key= lambda x:x.box.w)
        return temp[len(temp) / 2].box.w

    def _contain_space(self, first_char, second_char, median_width, percentage = 0.2):
        space_width = second_char.box.x - (first_char.box.x + first_char.box.w)
        if space_width >= median_width * percentage:
            return True
        return False

    def get_result(self):
        # Sort text blocks first, in x direction
        self.textChars.sort(key= lambda x:x.box.x)
        result = ''
        median_width = self._get_median_width()
        for i, textChar in enumerate(self.textChars):
            # Check if space exists
            if i > 0 and self._contain_space(self.textChars[i - 1], self.textChars[i], median_width,
                                             percentage=DEFAULT_SPACE_PERCENTAGE):
                result += ' '
            result += textChar.get_result()
        result += '\n'
        return result


class TextChar(object):
    """A class to represent a character image.
    """

    def __init__(self, img, box):
        self.img = img
        self.box = box
        self.char = None

    def recognize_char(self, knn):
        self.char = knn.recognize(normalize_image(self.img))

    def get_result(self):
        return self.char


class OcrKnn(object):
    """A class to represent the Knn of the system."""

    def __init__(self, classifications, flattened_images, n_neighbors = 5, weights = 'uniform', algorithm = 'auto'):
        self.classifications = classifications
        self.flattened_images = flattened_images
        self.knn = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm

    def create_and_train(self):
        """ Create a Knn instance of opencv, then train the knn
        """
        # self.knn = cv2.ml.KNearest_create()
        # self.knn.train(self.flattened_images, cv2.ml.ROW_SAMPLE, self.classifications)

        self.knn = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights,
                                                  algorithm=self.algorithm)
        self.knn.fit(self.flattened_images, self.classifications)

    def recognize(self, image):
        """ Return a tuple of the recognize character.
        """
        res = np.float32(image.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)))
        # ret, results, neighbors, dists = self.knn.findNearest(res, self.k)
        return str(chr(int(self.knn.predict(res))))
        # return str(chr(int(results[0][0])))


class Timer(object):

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.length = 0
        self.isTiming = False

    def Start(self):
        if self.isTiming:
            warnings.warn('This timer has been started. Consider End() it first.')
            return
        self.start_time = time()
        self.isTiming = True

    def End(self):
        if not self.isTiming:
            warnings.warn('This timer has not been started. Consider Start() it before.')
        self.isTiming = False
        self.end_time = time()
        self.length = self.end_time - self.start_time

    def GetCurrent(self):
        if self.isTiming:
            return time() - self.start_time
        return self.length


def Similarity(a, b):
    """
    Calculate the similarity of a and b. The denominator is len(b).
    :param a:
    :param b:
    :return:
    """
    s = difflib.SequenceMatcher(None, a, b)
    blocks = s.get_matching_blocks()
    # count all the similar
    count = 0
    match_string = ''
    for block in blocks:
        match_string += a[block.a:block.a+block.size]
        count += block.size
    # return difflib.SequenceMatcher(None, a, b).ratio() * 100.0
    if TESTING:
        print 'Longest matches: ' + match_string
        print 'Differences: '
        sys.stdout.writelines(list(difflib.Differ().compare(match_string, b)))
    return count * 100.0 / len(b)


# Global method
def normalize_image(gray_img):
    """Global method to get a normalize image for data set. Use for character image only."""
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized_img = cv2.resize(binary_img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    return resized_img


def nothing(x):
    pass