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
TESTING = False


# Image size to resized
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SPLIT_N = 6
BIN_N = 16

# Lower percentage of space w.r.t. medium width
DEFAULT_SPACE_PERCENTAGE = 0.5

# Enums
class Ocr(enum.Enum):
    PROJECTION = 1
    CONTOUR = 2
    COMBINE = 3
    BINARY_PROJECTION = 4

    FULL = 5

    # Text Document method
    DILATION = 6

    # Method to get features
    SIMPLE_10 = 7
    HOG = 8
    AVERAGE = 9
    SIMPLE_20 = 10
    SIMPLE_30 = 11
    SIMPLE_BIN_10 = 12
    SIMPLE_BIN_20 = 13
    SIMPLE_BIN_30 = 14
    SIMPLE_BIN_10_15 = 15
    SIMPLE_BIN_20_30 = 16
    SIMPLE_20_30 = 17
    SIMPLE_10_15 = 18
    SIMPLE_5 = 19
    SIMPLE_3_5 = 20
    SIMPLE_BIN_5 = 21
    SIMPLE_BIN_3_5 = 22

    # Method for


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

    def __init__(self, img, bin_img):
        """Initialize a text document."""
        self.img = img
        self.bin_img = bin_img
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
            crop_bin_img = self.bin_img[block_box.y: block_box.y + block_box.h, block_box.x: block_box.x + block_box.w]
            blocks.append(TextBlock(crop_img, crop_bin_img, block_box))

        if TESTING:
            text_image_copy = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            for l in block_boxes:
                cv2.rectangle(text_image_copy, (l.x, l.y), (l.x + l.w, l.y + l.h), (0, 255, 0), 1)
            cv2.imshow('test_blocks', text_image_copy)
            cv2.waitKey(0)

        # Assign text block inside:
        self.textBlocks = blocks

    def _get_text_block_by_dilation(self, params=None):

        # Default max components
        max_components = 4
        if params and 'max_components' in params:
            max_components = params['max_components']

        contours = self._find_components(self.bin_img, max_components=max_components)
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
        while count > max_components:
            dilated_image = self._dilate(input_img, size, iterations=iterations)
            # inverse the dilated image, since find contours only find black pixel
            if TESTING:
                cv2.imshow('dilated_image', dilated_image)
                cv2.waitKey(0)
            _, contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def __init__(self, img, bin_img, box):
        """
        Initialize a TextBlock
        :param img: The image
        :param box: A Box class for dimensions (x, y, w, h)
        :param method: method to extract lines from the block
        """
        self.img = img
        self.bin_img = bin_img
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
        elif method == Ocr.BINARY_PROJECTION:
            line_boxes = self._get_boxes_by_binary_projection(params)
        else:
            raise ValueError('Invalid method in get_text_lines: ' + str(method))

        for line_box in line_boxes:
            x, y, w, h = line_box.x, line_box.y, line_box.w, line_box.h
            crop_img = self.img[line_box.y: line_box.y + line_box.h, line_box.x: line_box.x + line_box.w]
            crop_bin_img = self.bin_img[line_box.y: line_box.y + line_box.h, line_box.x: line_box.x + line_box.w]
            width, height = crop_img.shape
            if width == 0 or height == 0:
                continue
            if TESTING:
                cv2.imshow('crop_img', crop_img)
                cv2.waitKey(0)
            lines.append(TextLine(crop_img, crop_bin_img, line_box))

        # Plot the process


        if TESTING:
            text_image_copy = self.img.copy()
            for l in line_boxes:
                cv2.rectangle(text_image_copy, (l.x, l.y), (l.x + l.w, l.y + l.h), (0, 255, 0), 1)
            cv2.imshow('test_lines', text_image_copy)
            cv2.waitKey(0)

        self.textLines = lines

    def _get_boxes_by_binary_projection(self, params):
        threshold = 0
        density_threshold = 0
        if (params and 'threshold' in params):
            threshold = params['threshold']
        if (params and 'density_threshold' in params):
            density_threshold = params['density_threshold']

        horizontal_projection = cv2.reduce(self.bin_img, 1, cv2.REDUCE_AVG)
        horizontal_projection_copy = horizontal_projection.copy()

        hist = horizontal_projection
        indices_low = horizontal_projection <= threshold
        indices_high = horizontal_projection > threshold
        hist[indices_low] = 0
        hist[indices_high] = 1

        ycoords = []
        y = 0
        count = 0
        is_space = False
        rows, cols = self.img.shape

        for i in range(rows):
            if not is_space:
                if not hist[i, 0]:
                    is_space = True
                    count = 1
                    y = i
            else:
                if hist[i, 0] and count > 0:
                    is_space = False
                    ycoords.append(y / count)
                else:
                    y += i
                    count += 1

        # Add final line
        ycoords.append(y / count)
        line_boxes = []
        for i in range(len(ycoords)):
            if i == 0: continue
            line_boxes.append(Box(0, ycoords[i - 1] + 1, cols, ycoords[i] - ycoords[i - 1] - 1))
        '''
        line_boxes = []
        for i in range(len(ycoords)):
            if i == len(ycoords) - 1:
                line_boxes.append(Box(0, ycoords[i] + 1, cols, rows - ycoords[i]))
            else:
                line_boxes.append(Box(0, ycoords[i] + 1, cols, ycoords[i + 1] - ycoords[i] - 1))
        '''
        if TESTING:
            temp = cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2BGR)
            for y in ycoords:
                cv2.line(temp, (0, y), (cols, y), (0, 255, 0), 2)

            plt.figure()
            plot1 = plt.subplot('221')
            plt.plot(horizontal_projection_copy)
            plt.subplot('222')

            # Rotate image
            # frows, cols = thresh_img.shape
            # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            # dst = cv2.warpAffine(thresh_img, M, (cols, rows))\
            plt.imshow(temp, cmap='gray')
            plt.show()
            plt.close()

        return line_boxes



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

    def __init__(self, img, bin_img, box):
        """
        Initialize and TextBlock
        :param img: The image
        :param box: A Box class for dimensions (x, y, w, h)
        :param method: method to extract lines from the block
        """
        self.img = img
        self.bin_img = bin_img
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

        if TESTING:
            line_image_copy = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            for c in character_boxes:
                cv2.rectangle(line_image_copy, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 255, 0), 1)
                cv2.imshow('find_characters', line_image_copy)
                cv2.waitKey(0)


        for character_box in character_boxes:
            crop_img = self.img[character_box.y: character_box.y + character_box.h,
                              character_box.x: character_box.x + character_box.w]
            crop_bin_img = self.bin_img[character_box.y: character_box.y + character_box.h,
                              character_box.x: character_box.x + character_box.w]

            characters.append(TextChar(crop_img, crop_bin_img, character_box))

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
        bin_img_copy = self.bin_img.copy()
        _, contours, _ = cv2.findContours(bin_img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    def __init__(self, img, bin_img, box):
        self.img = img
        self.bin_img = bin_img
        self.box = box
        self.char = None

    def recognize_char(self, knn, method):
        self.char = knn.recognize(self.img, method)

    def get_result(self):
        return self.char


class OcrKnn(object):
    """A class to represent the Knn of the system."""

    def __init__(self, labels, features, n_neighbors = 5, weights = 'uniform', algorithm = 'auto'):
        self.labels = labels
        self.features = features
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
        self.knn.fit(self.features, self.labels)

    def recognize(self, image, method):
        """ Return a tuple of the recognize character. Input is the gray-image itself.
        """
        res = None
        if method == Ocr.SIMPLE_3_5:
            res = preprocess_simple(image, (3, 5))
        elif method == Ocr.SIMPLE_5:
            res = preprocess_simple(image, (5, 5))
        elif method == Ocr.SIMPLE_10:
            res = preprocess_simple(image, (10, 10))
        elif method == Ocr.SIMPLE_20:
            res = preprocess_simple(image, (20, 20))
        elif method == Ocr.SIMPLE_30:
            res = preprocess_simple(image, (30, 30))
        elif method == Ocr.SIMPLE_20_30:
            res = preprocess_simple(image, (20, 30))
        elif method == Ocr.SIMPLE_10_15:
            res = preprocess_simple(image, (10, 15))
        elif method == Ocr.SIMPLE_BIN_3_5:
            res = preprocess_simple_binary(image, (3, 5))
        elif method == Ocr.SIMPLE_BIN_5:
            res = preprocess_simple_binary(image, (5, 5))
        elif method == Ocr.SIMPLE_BIN_10:
            res = preprocess_simple_binary(image, (10, 10))
        elif method == Ocr.SIMPLE_BIN_20:
            res = preprocess_simple_binary(image, (20, 20))
        elif method == Ocr.SIMPLE_BIN_30:
            res = preprocess_simple_binary(image, (30, 30))
        elif method == Ocr.SIMPLE_BIN_20_30:
            res = preprocess_simple_binary(image, (20, 30))
        elif method == Ocr.SIMPLE_BIN_10_15:
            res = preprocess_simple_binary(image, (10, 15))
        elif method == Ocr.HOG:
            res = preprocess_hog(image)
        elif method == Ocr.AVERAGE:
            res = preprocess_average(image)
        # res = np.float32(image.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)))
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
        print '\n'
    return count * 100.0 / len(b)


# Global method
def preprocess_simple(gray_img, size):
    """Simply flatten a image to WIDTH * HEIGHT features"""
    # blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # binary_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # resized_img = cv2.resize(binary_img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    # return resized_img.reshape((1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH))
    width, height = size
    resized_img = cv2.resize(gray_img, size)
    return np.float32(resized_img.reshape(-1, width * height) / 255.0)


def preprocess_simple_binary(gray_img, size):
    width, height = size
    blur_img = cv2.medianBlur(gray_img, 5)
    _, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, morph_kernel)
    resized_img = cv2.resize(morph_img, size)
    return np.float32(resized_img.reshape(-1, width * height))


def preprocess_hog(gray_img):
    """Use hog to calculate the features"""

    resized_img = cv2.resize(gray_img, ((RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)))
    gx = cv2.Sobel(resized_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(resized_img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    bin_n = BIN_N
    bin = np.int32(bin_n * ang / (2 * np.pi))

    bin_cells = bin[0:10, 0:10], bin[0:10, 10:20], bin[0:10, 20:30], \
                bin[10:20, 0:10], bin[10:20, 10:20], bin[10:20, 20:30]
    mag_cells = mag[0:10, 0:10], mag[0:10, 10:20], mag[0:10, 20:30], \
                mag[10:20, 0:10], mag[10:20, 10:20], mag[10:20, 20:30]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps

    return np.float32(hist.reshape(-1, BIN_N * SPLIT_N))

def preprocess_average(gray_img):
    return None

def preprocess_image(gray_img, skew_correction=False):
    """Return a tuple of original gray image after skew and the binary image after skew."""
    blur_img = cv2.medianBlur(gray_img, 5)
    thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, morph_kernel)

    # Skew correction
    if skew_correction:
        pts = cv2.findNonZero(morph_img)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        center, _, angle = rect
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rows, cols = morph_img.shape

        rotated_gray = cv2.warpAffine(gray_img, m, (cols, rows))
        rotated_bin = cv2.warpAffine(morph_img, m, (cols, rows))
        return rotated_gray, rotated_bin

    return gray_img, morph_img


def nothing(x):
    pass