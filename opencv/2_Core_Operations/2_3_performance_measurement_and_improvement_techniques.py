import numpy as np
import cv2


def main():
    img1 = cv2.imread('../samples/hammerhead.jpg')

    e1 = cv2.getTickCount()
    for i in range(5,49,2):
        img1 = cv2.medianBlur(img1, i)

    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print(t)

    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
