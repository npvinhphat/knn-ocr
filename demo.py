"""
Demo module to get the result from camera.
"""

import os
import cv2
import ocr_knn
import numpy as np
import recognize

DATA_PATH = recognize.DATA_PATH
TESTING = True

DATA = ('11_fonts_gray_10', ocr_knn.Ocr.SIMPLE_10)
K = 3
WEIGHT = 'distance'
ALGORITHM = 'brute'

def get_image_from_camera():
    cap = cv2.VideoCapture(0)
    result = None
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            result = frame
            break
    cap.release()
    cv2.destroyAllWindows()
    return result

def demo_test(test):
    test_name = test.test_name
    test_result = test.test_result
    data_name = test.data_name
    n_neighbors = test.n_neighbors
    weights = test.weights
    algorithm = test.algorithm
    ocr_method = test.ocr_method
    test_image = test.test_image

    image = test_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray, bin = ocr_knn.preprocess_image(gray)

    # Find all the blocks, lines, then characters inside the document
    textDocument = ocr_knn.TextDocument(gray, bin)
    textDocument.get_text_blocks(method=ocr_knn.Ocr.DILATION, params={'max_components': 1})
    for textBlock in textDocument.textBlocks:
        textBlock.get_text_lines(method=ocr_knn.Ocr.BINARY_PROJECTION, params={'threshold': 25})
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

    return recognized_result

def main():
    # Get image from camera
    original_img = get_image_from_camera()
    data_name, method = DATA
    # rotated_img = ocr_knn.preprocess_image(original_img)
    # cv2.imshow('rotate', rotated_img)
    # cv2.waitKey(0)
    test = recognize.Test('camera_test', None, data_name, K, WEIGHT, ALGORITHM, method, test_image=original_img)
    recognized_result = demo_test(test)
    print recognized_result


if __name__ == '__main__':
    main()