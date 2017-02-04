"""
Recognize all the characters inside a given image.
"""

import cv2
import ocr_knn
import numpy as np
import os
import openpyxl
from sklearn import metrics

# Modify here to use different data, a tuple of data and method for ta
DATAS = [('11_fonts_gray_10', ocr_knn.Ocr.SIMPLE_10)]
DATA_PATH = 'train_data'

# Modify here to use different test, a tuple of tess_name and the result file
TESTS_12FONTS = [('gill-sans.png', 'abcxyz.txt'),
                 ('calibri.png', 'abcxyz.txt'),
                 ('arial.png', 'abcxyz.txt'),
                 ('times-new-roman.png', 'abcxyz.txt'),
                 ('rockwell.png', 'abcxyz.txt'),
                 ('lucida-sans.png', 'abcxyz.txt'),
                 ('adabi-mt.png', 'abcxyz.txt'),
                 ('tw-cen-mt.png','abcxyz.txt'),
                 ('cambria.png','abcxyz.txt'),
                 ('news-gothic-mt.png', 'abcxyz.txt'),
                 ('candara.png','abcxyz.txt')]

TESTS_12FONTS_DIFF = [('andale-mono.png', 'abcxyz.txt'),
                     ('arial-hebrew.png', 'abcxyz.txt'),
                     ('bell-mt.png', 'abcxyz.txt'),
                     ('cambria-math.png', 'abcxyz.txt'),
                     ('century.png', 'abcxyz.txt'),
                     ('comic-sans.png', 'abcxyz.txt'),
                     ('gill-sans-mt.png', 'abcxyz.txt'),
                     ('helvetica.png','abcxyz.txt'),
                     ('lao-sangam-mn.png','abcxyz.txt'),
                     ('optima.png', 'abcxyz.txt'),
                     ('skia.png','abcxyz.txt')]

TESTS = TESTS_12FONTS_DIFF + TESTS_12FONTS + TESTS_12FONTS + TESTS_12FONTS_DIFF + TESTS_12FONTS + TESTS_12FONTS_DIFF
TESTS = TESTS_12FONTS_DIFF
# TESTS = [('simple_lowercase.png', None)]
TEST_PATH = 'tests'

"""Parameters"""
IGNORE_SPACE = True
TESTING = False
KS = [3, 4, 5]
WEIGHTS = ['distance']
ALGORITHMS = ['brute']

# Output file
OUTPUT_FILE = 'output.txt'
OUTPUT_XLSX = 'data.xlsx'

# Global data for getting confusion matrix
x_true = []
x_predict = []

class Test(object):

    def __init__(self, test_name, test_result, data_name, n_neighbors, weights, algorithm, ocr_method, test_image=None):
        self.test_name = test_name
        self.test_result = test_result
        self.data_name = data_name
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.ocr_method = ocr_method
        self.test_image = test_image

    def __str__(self):
        test_name = 'Test: %s\n' % self.test_name
        test_result = 'Result: %s\n' % self.test_result
        data_name = 'Data: %s\n' % self.data_name
        n_neighbors = 'K: %d\n' % self.n_neighbors
        weights = 'Weight: %s\n' % self.weights
        algorithm = 'Algorithm: %s\n' % self.algorithm
        ocr_method = 'OCR Method: %s\n' % self.ocr_method

        return test_name + test_result + n_neighbors + weights + algorithm + ocr_method

class Result(object):

    def __init__(self, overall_time, total_recognize_time, total_build_time, character_count, total_accuracy,
                 recognized_result):
        self.overall_time = overall_time
        self.total_recognize_time = total_recognize_time
        self.total_build_time = total_build_time
        self.character_count = character_count
        self.total_accuracy = total_accuracy
        self.recognized_result = recognized_result

    def __str__(self):
        recognized_result = 'Recognized: \"%s\"\n' % self.recognized_result
        overall_time = 'Overall Time: %s\n' % str(self.overall_time)
        total_recognize_time = 'Total Recognize Time: %s\n' % str(self.total_recognize_time)
        total_recognize_time = 'Total Build Time: %s\n' % str(self.total_build_time)
        character_count = 'Character Count: %d\n' % self.character_count
        total_accuracy = 'Total Accuracy: %s\n' % str(self.total_accuracy)

        return recognized_result + overall_time + total_recognize_time + character_count + total_accuracy

def get_result(test):
    """
    Get the result in stored it in result class.
    """

    # Retrieve data from test
    test_name = test.test_name
    test_result = test.test_result
    data_name = test.data_name
    n_neighbors = test.n_neighbors
    weights = test.weights
    algorithm = test.algorithm
    ocr_method = test.ocr_method
    test_image = test.test_image

    # First timer for calculating result
    all_timer = ocr_knn.Timer()
    all_timer.Start()

    image = test_image
    # If the image is not provided, use from the test_name
    if image is None:
        image = cv2.imread(os.path.join(TEST_PATH, test_name))
    # If it's still not provided, return error
    if image is None:
        raise ValueError('Image %s Not Found!' % image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray, bin = ocr_knn.preprocess_image(gray)
    # gray = ocr_knn.preprocess_image(image)

    # Find all the blocks, lines, then characters inside the document
    textDocument = ocr_knn.TextDocument(gray, bin)
    textDocument.get_text_blocks(method=ocr_knn.Ocr.DILATION, params={'max_components': 1})
    for textBlock in textDocument.textBlocks:
        textBlock.get_text_lines(method=ocr_knn.Ocr.BINARY_PROJECTION, params={'threshold':0})
        for textLine in textBlock.textLines:
            textLine.get_text_chars(method=ocr_knn.Ocr.COMBINE)

    # Load the data
    labels = np.loadtxt(os.path.join(DATA_PATH, data_name, 'labels.txt'), np.float32)
    features = np.loadtxt(os.path.join(DATA_PATH, data_name, 'features.txt'), np.float32)

    # Reshape numpy array to 1-d, necessary to pass to call to train
    labels = labels.reshape((labels.size, 1)).ravel()

    # Create OcrKnn instance
    ocrKnn = ocr_knn.OcrKnn(labels, features, n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    total_build_time = 0
    build_timer = ocr_knn.Timer()
    build_timer.Start()
    ocrKnn.create_and_train()
    build_timer.End()
    total_build_time += build_timer.GetCurrent()

    # Recognize each character and store in each text char
    total_recognize_time = 0
    count = 0
    recognize_timer = ocr_knn.Timer()
    for textBlock in textDocument.textBlocks:
        for textLine in textBlock.textLines:
            for textChar in textLine.textChars:
                recognize_timer.Start()
                textChar.recognize_char(ocrKnn, method=ocr_method)
                recognize_timer.End()
                total_recognize_time += recognize_timer.GetCurrent()
                count += 1

    # In case the program needs to debug the result
    if TESTING:
        document_img = cv2.cvtColor(textDocument.img, cv2.COLOR_GRAY2BGR)
        for textBlock in textDocument.textBlocks:
            cv2.rectangle(document_img, (textBlock.box.x, textBlock.box.y),
                          (textBlock.box.x + textBlock.box.w, textBlock.box.y + textBlock.box.h), (0, 255, 0), 1)
        cv2.imshow('document_img', document_img)
        cv2.waitKey(0)

        for textBlock in textDocument.textBlocks:
            block_img = cv2.cvtColor(textBlock.img, cv2.COLOR_GRAY2BGR)
            for textLine in textBlock.textLines:
                cv2.rectangle(block_img, (textLine.box.x, textLine.box.y),
                              (textLine.box.x + textLine.box.w, textLine.box.y + textLine.box.h), (0, 255, 0), 1)
            cv2.imshow('block_img', block_img)
            cv2.waitKey(0)

            for textLine in textBlock.textLines:
                _, thresh_line = cv2.threshold(textLine.img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
                line_img = cv2.cvtColor(thresh_line, cv2.COLOR_GRAY2BGR)
                for textChar in textLine.textChars:
                    cv2.rectangle(line_img, (textChar.box.x, textChar.box.y),
                                  (textChar.box.x + textChar.box.w, textChar.box.y + textChar.box.h), (0, 255, 0), 1)
                cv2.imshow('line_img', line_img)
                cv2.waitKey(0)


    # Get the result
    recognized_result = textDocument.get_result()

    # Check if the result exists? if yes, use it
    accuracy = None
    if test_result and os.path.isfile(os.path.join(TEST_PATH, test_result)):
        path_to_result = os.path.join(TEST_PATH, test_result)
        with open(path_to_result, 'r') as result_file:
            correct_string = result_file.read()
            print 'The correct string is: \n\"' + correct_string + '\"'
            # Check to see how many percentage is correct, using difflib
            # Check if ignore the space
            if IGNORE_SPACE:
                accuracy = ocr_knn.Similarity(''.join(recognized_result.split()), ''.join(correct_string.split()))
                global x_true, x_predict
                x_true = x_true + list(''.join(recognized_result.split()))
                x_predict = x_predict + list(''.join(correct_string.split()))
                print 'The percentage of similarity is (without spaces): %s' % str(accuracy)
            else:
                accuracy = ocr_knn.Similarity(recognized_result, correct_string)
                print 'The percentage of similarity is: %s' + str(accuracy)

    cv2.destroyAllWindows()

    all_timer.End()

    print 'Total recognize time: %f seconds' % total_recognize_time
    print 'Overall time: %f seconds' % all_timer.GetCurrent()

    result = Result(all_timer.GetCurrent(), total_recognize_time, total_build_time, count, accuracy, recognized_result)
    return result

# Main
def main():
    # Analytical to get tests and results file
    tests = []
    results = []
    for test_name, test_result in TESTS:
        for k in KS:
            for weight in WEIGHTS:
                for algorithm in ALGORITHMS:
                    for data in DATAS:
                        data_name, method = data
                        print '----------------------------'
                        test = Test(test_name, test_result, data_name, k, weight, algorithm, method)
                        result = get_result(test)

                        tests.append(test)
                        results.append(result)
                        print '\n%s\n%s\n' % (str(test), str(result))
                        print '----------------------------'

    with open(OUTPUT_FILE, 'a') as f:
        for test, result in zip(tests, results):
            # print result.total_accuracy
            f.write('\n%s\n%s\n' % (str(test), str(result)))

    # Generate excel file
    wb = openpyxl.Workbook()
    ws = wb.active
    rows = [['Test Name', 'Data', 'K', 'Weight', 'Algorithm', 'Method', 'Overall Time', 'Total Recognize Time', 'Total Build Time',
             'Character Count', 'Accuracy']]
    for test, result in zip(tests, results):
        new_list = [test.test_name, test.data_name, test.n_neighbors, test.weights, test.algorithm,
                    str(test.ocr_method),
                    result.overall_time, result.total_recognize_time, result.total_build_time, result.character_count, result.total_accuracy]
        rows.append(new_list)
    for row in rows:
        ws.append(row)
    wb.save('data.xlsx')

    # Generate confusion matrix
    valid_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U',
                   'V', 'W', 'X', 'Y', 'Z',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u',
                   'v', 'w', 'x', 'y', 'z',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    my_dict = dict()
    for c in valid_chars:
        my_dict[c] = 0
    for a, b in zip(x_true, x_predict):
        if a == b:
            my_dict[a] += 1

    my_dict = sorted(my_dict.items())
    list1 = [key for (key, value) in my_dict]
    list2 = [value for (key, value) in my_dict]
    '''
    print x_true
    print x_predict
    confusion_matrix = metrics.confusion_matrix(x_true, x_predict, labels=valid_chars)
    '''
    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    '''
    ws2.append(valid_chars)
    for row in confusion_matrix:
        print row
        ws2.append(row.tolist())
    '''
    ws2.append(list1)
    ws2.append(list2)
    wb2.save('confusion.xlsx')




if __name__ == '__main__':
    main()