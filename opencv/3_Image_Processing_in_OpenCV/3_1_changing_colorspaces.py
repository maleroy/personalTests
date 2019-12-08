import numpy as np
import cv2


# for listing all flags, see [i for i in dir(cv2) if i.startswith('COLOR_')]

BLUE = np.uint8([[[255, 0, 0]]])
HSV_BLUE = cv2.cvtColor(BLUE, cv2.COLOR_BGR2HSV)

GREEN = np.uint8([[[0, 255, 0]]])
HSV_GREEN = cv2.cvtColor(GREEN, cv2.COLOR_BGR2HSV)

RED = np.uint8([[[0, 0, 255]]])
HSV_RED = cv2.cvtColor(RED, cv2.COLOR_BGR2HSV)

YELLOW = np.uint8([[[0, 255, 255]]])
HSV_YELLOW = cv2.cvtColor(YELLOW, cv2.COLOR_BGR2HSV)

def main():
    cap = cv2.VideoCapture(0)

    while(1):

        # Take each frame
        _, frame = cap.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of color in HSV
        color = HSV_YELLOW
        lower_color = np.array([color[0, 0, 0]-10, 50, 50])
        upper_color = np.array([color[0, 0, 0]+10, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()