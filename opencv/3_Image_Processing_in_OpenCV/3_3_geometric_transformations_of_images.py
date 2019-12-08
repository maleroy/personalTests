import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('../samples/hammerhead.jpg')
    img_gray = cv2.imread('../samples/hammerhead.jpg', 0)

    ### SCALING
    # Preferable interpolation methods are cv2.INTER_AREA for shrinking
    # and cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming.
    # By default, interpolation method used is cv2.INTER_LINEAR for all resizing purposes
    res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #OR
    height, width = img.shape[:2]
    res = cv2.resize(img, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)    

    ### TRANSLATION
    rows, cols = img_gray.shape

    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img_gray, M, (cols, rows))

    cv2.imshow('img_gray', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### ROTATION
    # Contrary to standard 2D rotation [[cos(t), -sin(t)], [sin(t), cos(t)]],
    # OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer.
    # Modified transformation matrix is given by
    # [[alpha, beta, (1-alpha)*c_x - beta*c_y], [-beta, alpha, beta*c_x + (1-alpha)*c_y]]
    rows, cols = img_gray.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    dst = cv2.warpAffine(img_gray, M, (cols, rows))

    cv2.imshow('img_gray', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # AFFINE TRANSFORMATION
    # In affine transformation, all parallel lines in the original image will still be parallel in the output image.
    # To find the transformation matrix, we need three points from input image and their corresponding locations in output image.
    # Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    dst = cv2.warpAffine(img, M, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


if __name__ == '__main__':
    main()