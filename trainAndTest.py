""" trainAndTest.py
"""

import cv2
import numpy as np
import operator
import os
import sys
import gflags
import difflib
from matplotlib import pyplot as plt

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('k', None, 'The K value for KNN training.')
gflags.DEFINE_boolean('ignore_space', False, 'Check if you want to ignore space checking between words.')
gflags.DEFINE_string('path_to_result', None, 'The path to the result file (.txt)')
gflags.DEFINE_string('path_to_test', 'tests/andale_mono.png', 'The path to the test file (image file)')

# Mark test flag as required
gflags.MarkFlagAsRequired('path_to_test')

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 300

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
SPACE_PERCENTAGE = 0.6

class ContourWithData():

    def __init__(self):
        self.npaContour = None           # contour
        self.boundingRect = None         # bounding rect for contour
        self.intRectX = 0                # bounding rect top left corner x location
        self.intRectY = 0                # bounding rect top left corner y location
        self.intRectWidth = 0            # bounding rect width
        self.intRectHeight = 0           # bounding rect height
        self.fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA:
            return False        # much better validity checking would be necessary
        return True

    def isSplitable(self):
        # return True if the width is larger than the height
        return self.intRectWidth > self.intRectHeight

    def __cmp__(self, other):
        return cmp(self.intRectX, other.intRectX)

class Character():

    def __init__(self):
        self.contourWithData = None
        self.value = None

def _ContainSpace(firstCharacter, secondCharacter, threshold=0.2):
    averageWidth = (firstCharacter.contourWithData.intRectWidth + secondCharacter.contourWithData.intRectWidth) / 2.0
    spaceWidth = secondCharacter.contourWithData.intRectX - firstCharacter.contourWithData.intRectX - firstCharacter.contourWithData.intRectWidth
    if spaceWidth > (1 - threshold) * averageWidth and spaceWidth < (1 + threshold) * averageWidth:
        return True
    return False

def _Similarity(a, b):
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
    print 'Longest matches: ' + match_string
    print 'Differences: '
    sys.stdout.writelines(list(difflib.Differ().compare(match_string, b)))
    return count * 100.0 / len(b)

class CharacterLine():

    def __init__(self):
        self.characters = []
        self.top = sys.maxint
        self.bottom = 0

    def addNewCharacter(self, character):
        """Add a new character to this character line."""
        self.characters.append(character)
        # Update this whole line
        self.top = min(self.top, character.contourWithData.intRectY)
        self.bottom = max(self.bottom, character.contourWithData.intRectY + character.contourWithData.intRectHeight)

    def isInside(self, character):
        """Check if this character belongs to this line."""
        characterTop = character.contourWithData.intRectY
        characterBottom = character.contourWithData.intRectY + character.contourWithData.intRectHeight
        # Check if two intervals collapsed
        return max(characterTop, self.top) < min(characterBottom, self.bottom)

    def __str__(self):
        # Sort all the characters inside first
        self.characters.sort(key= operator.attrgetter('contourWithData'))
        # The result
        result = ''
        # Create the string
        lastCharacter = None
        for character in self.characters:
            # Check if there is space between two consecutive characters
            if lastCharacter is not None and _ContainSpace(lastCharacter, character, threshold= SPACE_PERCENTAGE):
                result = result + ' '
            result = result + character.value
            lastCharacter = character
        # Add a newline
        result = result + '\n'
        return result

class Document():

    def __init__(self, characterLines):
        self.characterLines = sorted(characterLines, key= operator.attrgetter('top'))

    def convertToSingleString(self):
        result = ''
        for characterLine in self.characterLines:
            result = result + str(characterLine)
        return result

def main(argv):

    # Import flags
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError, e:
        print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
        sys.exit(1)

    # If FLAGS.k is not given, use 1 instead
    if not FLAGS.k:
        FLAGS.k = 3

    # Start the program
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    # Reshape numpy array to 1-d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    # Instantiate KNN object
    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread(FLAGS.path_to_test)          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    # imgBlurred = cv2.GaussianBlur(imgGray, (3, 3), 0)                    # blur
    imgBlurred = cv2.medianBlur(imgGray, 3)
    cv2.imshow('Blurred Img', imgBlurred)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological gradient
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient_img = cv2.morphologyEx(imgThresh, cv2.MORPH_GRADIENT, morph_kernel)
    cv2.imshow('gradient_img', gradient_img)

    # Binary
    # _, binary_img = cv2.threshold(gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary_img = cv2.adaptiveThreshold(gradient_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow('binary_img', binary_img)

    imgThreshCopy = gradient_img.copy()
    imgThreshCopy2 = gradient_img.copy()

    cv2.imshow('imgThresCopy', imgThreshCopy)

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data

    # Create valid contours
    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            # if contourWithData.intRectWidth > contourWithData.intRectHeight:
            if contourWithData.isSplitable():
                pass
                # print str(contourWithData.intRectWidth) + ' ' + str(contourWithData.intRectHeight)
                # imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                #          contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]
                # slice = cv2.reduce(imgROI, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S) / 255
                # print slice
                # cv2.imshow('imgROI', imgROI)
                # plt.plot(slice[0])
                # plt.show()
                # cv2.waitKey(0)
        validContoursWithData.append(contourWithData)       # if so, append to valid contour list

    # validContoursWithData.sort(key = operator.attrgetter("intRectY", "intRectX"))         # sort contours from top to bottom, left to right
    validContoursWithData.sort(key=operator.attrgetter("intRectX", "intRectY"))

    characters = []

    for contourWithData in validContoursWithData:

        # Draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),
                      (0, 255, 0),
                      2)

        # Crop the character out of threshold image
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        # Resize the image for consistency with the database
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # Flatten the image into 1-d numpy array
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        # Convert numpy array from int to float32
        npaROIResized = np.float32(npaROIResized)

        # Call K-NN function find_nearest
        ret, results, neighbors, dists = kNearest.findNearest(npaROIResized, k = FLAGS.k)

        # Get the character from results
        strCurrentChar = str(chr(int(results[0][0])))

        # Convert from float to characters
        neighbors_char = [str(chr(int(neighbor))) for neighbor in neighbors[0]]
        ret_char = str(chr(int(ret)))

        # Print the result with their neighbors
        print('ret: ', ret)
        print('ret_char: ', ret_char)
        print('results: ', results)
        print('neighbors: ', neighbors)
        print('neighbors_char: ', neighbors_char)
        print('dists:', dists)
        cv2.imshow('imgROI', imgROI)
        cv2.imshow('imgTestingNumbers', imgTestingNumbers)
        cv2.waitKey(0)

        # Create a new Character, and add to all characters list
        currChar = Character()
        currChar.contourWithData = contourWithData
        currChar.value = strCurrentChar
        characters.append(currChar)

    # Create characters lines, end the loop when all the characters has been resolved
    characterLines = []
    for character in characters:
        # Check if this character is inside any given lines
        check = False
        for characterLine in characterLines:
            if characterLine.isInside(character):
                characterLine.addNewCharacter(character)
                check = True
                break

        # If this character doesn't belong to any lines, add a new line
        if not check:
            characterLine = CharacterLine()
            characterLine.addNewCharacter(character)
            characterLines.append(characterLine)

    # Sort all the lines by their top
    characterLines.sort(key= operator.attrgetter('top'))

    # Create a document file from all the character lines
    document = Document(characterLines)

    # Create a string for the final result
    resulting_string = document.convertToSingleString()

    # Print the final result
    print 'The resulting string is: \n\"' + resulting_string + '\"\n'

    # In case there are result file, use the result file to check with the result string
    if FLAGS.path_to_result:
        with open(FLAGS.path_to_result, 'r') as result_file:
            correct_string = result_file.read()
            print 'The correct string is: \n\"' + correct_string + '\"'
            # Check to see how many percentage is correct, using difflib
            # Check if ignore the space
            if FLAGS.ignore_space:
                print 'The percentage of similarity is (without spaces): ' + str(_Similarity(resulting_string.replace(' ', ''), correct_string.replace(' ', '')))
            else:
                print 'The percentage of similarity is: ' + str(_Similarity(resulting_string, correct_string))

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory


if __name__ == '__main__':
    main(sys.argv)









