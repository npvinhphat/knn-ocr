"""
Playground
"""

import cv2
import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import datetime

def main(unused_argv):
    # Read the image
    original_img = cv2.imread('tests/test_sans_serif.png')

    # Downsize the image for processing
    down_img = cv2.pyrDown(original_img)
    cv2.imshow('down_img', down_img)

    # Convert to gray
    gray_img = cv2.cvtColor(down_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_img', gray_img)

    # Morphological gradient
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient_img = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, morph_kernel)
    cv2.imshow('gradient_img', gradient_img)

    # Binary
    _, binary_img = cv2.threshold(gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary_img = cv2.adaptiveThreshold(gradient_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('binary_img', binary_img)

    # Connect horizontally connected regions
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, morph_kernel)
    cv2.imshow('connected_img', connected_img)

    # Create global mask
    height, width = binary_img.shape

    vertical_projection =  np.sum(binary_img, axis=1) / 255
    plt.close()

    # Find contours
    _, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    good_contours = []
    with PdfPages('thesis-finding.pdf') as pdf:
    # Filter contour
        for idx, contour in enumerate(contours):

            # Get the rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            print x, y, w, h

            # Create a mask to fill the contour
            mask = np.zeros(binary_img.shape, np.uint8)
            cv2.drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED)

            # Calculate the ratio and see if it fit
            ratio = 1.0 * cv2.countNonZero(mask) / (w * h)
            if ratio > .45 and w > 8 and h > 8:
                good_contours.append(contour)
                cv2.rectangle(down_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Sketch projection
                roi = binary_img[y: y + h, x: x + w]
                print roi.shape
                horizontal_projection = np.sum(roi, axis=0) / 255

                # Print them
                plt.figure()
                plot1 = plt.subplot(211)
                plt.plot(horizontal_projection)
                plt.subplot(212, sharex=plot1)
                plt.imshow(roi, cmap='Greys')
                pdf.savefig()
                plt.close()

        # Create pdf info
        d = pdf.infodict()
        d['Title'] = 'OCR-KNN Finding'
        d['Author'] = 'Phat P.V. Nguyen'
        d['Subject'] = 'Text with horizontal projection'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

    # Show the final image once more
    cv2.imshow('down_img_2', down_img)

    cv2.waitKey(0)

if __name__ == '__main__':
    main(sys.argv)