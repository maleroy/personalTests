import numpy as np
import cv2
import matplotlib.pyplot as plt


IMAGE = '../samples/hammerhead.jpg'


def main():
    # Load a color image in color (cv2.IMREAD_COLOR, default)
    img_color = cv2.imread(IMAGE, 1)
    # Load an image in grayscale (cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread(IMAGE, 0)
    # Load an image as such, with alpha (cv2.IMREAD_UNCHANGED)
    img_alpha = cv2.imread(IMAGE, -1)

    # By default, window not resizeable as using cv2.WINDOW_AUTOSIZE
    # cv2.WINDOW_NORMAL makes it possible to do so
    cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Color Image', img_color)

    cv2.namedWindow('Grayscale Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Grayscale Image', img_gray)
    
    cv2.namedWindow('Image as such', cv2.WINDOW_NORMAL)
    cv2.imshow('Image as such', img_alpha)

    my_key = cv2.waitKey(0)
    # ESC key to kill all windows
    if my_key == 27:
        cv2.destroyAllWindows()
    # 's' key to save then kill all windows
    elif my_key == ord('s'):
        cv2.imwrite('../samples/hammerhead_gray.jpg', img_gray)
        cv2.destroyAllWindows()

    plt.imshow(img_gray, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()