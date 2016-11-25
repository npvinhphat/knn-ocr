"""
Recognize all the characters inside a given image.
"""

import cv2
import ocr_knn
import numpy as np
import os

# Modify here to use different data
DATA_NAME = 'ten_fonts'
DATA_PATH = 'train_data'

# Modify here to use different test
TEST_NAME = 'simple_document.png'
TEST_RESULT = 'simple_document_result.txt'
TEST_PATH = 'tests'

# Use this to ignore space
IGNORE_SPACE = True

# Main
def main():
    # First timer for calculating result
    all_timer = ocr_knn.Timer()
    all_timer.Start()

    image = cv2.imread(os.path.join(TEST_PATH, TEST_NAME))
    if image is None:
        raise ValueError('Image %s Not Found!' % image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    # Find all the blocks, lines, then characters inside the document
    textDocument = ocr_knn.TextDocument(gray)
    textDocument.get_text_blocks(method=ocr_knn.Ocr.DILATION)
    for textBlock in textDocument.textBlocks:
        textBlock.get_text_lines(method=ocr_knn.Ocr.PROJECTION)
        for textLine in textBlock.textLines:
            textLine.get_text_chars(method=ocr_knn.Ocr.COMBINE)

    # Load the data
    classifications = np.loadtxt(os.path.join(DATA_PATH, DATA_NAME, 'classifications.txt'), np.float32)
    flattened_images = np.loadtxt(os.path.join(DATA_PATH, DATA_NAME, 'flattened_images.txt'), np.float32)

    # Reshape numpy array to 1-d, necessary to pass to call to train
    classifications = classifications.reshape((classifications.size, 1)).ravel()

    # Create OcrKnn instance
    ocrKnn = ocr_knn.OcrKnn(classifications, flattened_images, n_neighbors=3, weights='uniform', algorithm='ball_tree')
    ocrKnn.create_and_train()

    # Recognize each character and store in each text char
    total_recognize_time = 0
    recognize_timer = ocr_knn.Timer()
    for textBlock in textDocument.textBlocks:
        for textLine in textBlock.textLines:
            for textChar in textLine.textChars:
                recognize_timer.Start()
                textChar.recognize_char(ocrKnn)
                recognize_timer.End()
                total_recognize_time += recognize_timer.GetCurrent()

    # Get the result
    result = textDocument.get_result()
    print result

    # Check if the result exists? if yes, use it
    if TEST_RESULT and os.path.isfile(os.path.join(TEST_PATH, TEST_RESULT)):
        path_to_result = os.path.join(TEST_PATH, TEST_RESULT)
        with open(path_to_result, 'r') as result_file:
            correct_string = result_file.read()
            print 'The correct string is: \n\"' + correct_string + '\"'
            # Check to see how many percentage is correct, using difflib
            # Check if ignore the space
            if IGNORE_SPACE:
                print 'The percentage of similarity is (without spaces): ' + \
                      str(ocr_knn.Similarity(result.replace(' ', ''), correct_string.replace(' ', '')))
            else:
                print 'The percentage of similarity is: ' + str(ocr_knn.Similarity(result, correct_string))

    cv2.destroyAllWindows()

    all_timer.End()
    print 'Total recognize time: %f seconds' % total_recognize_time
    print 'Overall time: %f seconds' % all_timer.GetCurrent()


if __name__ == '__main__':
    main()