import numpy as np
import cv2

#TASK = 'capture'
#TASK = 'play'
TASK = 'save'

def main():
    # Argument of object can be device index (as many as we want)
    # or the name of a video file as it will be shown later
    if TASK == 'capture':
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cap.open()

        print(0, cap.get(0))  # Current position of the video file in milliseconds or video capture timestamp.
        print(1, cap.get(1))  # 0-based index of the frame to be decoded/captured next.
        print(2, cap.get(2))  # Relative position of the video file: 0 - start of the film, 1 - end of the film.
        print(3, cap.get(3))  # Width of the frames in the video stream.
        print(4, cap.get(4))  # Height of the frames in the video stream.
        print(5, cap.get(5))  # Frame rate.
        print(6, cap.get(6))  # 4-character code of codec.
        print(7, cap.get(7))  # Number of frames in the video file.
        print(8, cap.get(8))  # Format of the Mat objects returned by retrieve().
        print(9, cap.get(9))  # Backend-specific value indicating the current capture mode.
        print(10, cap.get(10))  # Brightness of the image (only for cameras).
        print(11, cap.get(11))  # Contrast of the image (only for cameras).
        print(12, cap.get(12))  # Saturation of the image (only for cameras).
        print(13, cap.get(13))  # Hue of the image (only for cameras).
        print(14, cap.get(14))  # Gain of the image (only for cameras).
        print(15, cap.get(15))  # Exposure (only for cameras).
        print(16, cap.get(16))  # Boolean flags indicating whether images should be converted to RGB.
        print(17, cap.get(17))  # The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        print(18, cap.get(18))  # The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        print(19, cap.get(19))  # Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        print(20, cap.get(20))  # The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently)
        print(21, cap.get(21))  # Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently)

        while(True):
            # Capture frame-by-frame, returns True if frame read correctly
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    elif TASK == 'play':
        vid = cv2.VideoCapture('../samples/tree.avi')
        while(vid.isOpened()):
            ret, frame = vid.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

    elif TASK == 'save':
        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        # In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('../samples/output.avi',fourcc, 20.0, (1280, 720))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                frame = cv2.flip(frame, 0)

                # write the flipped frame
                out.write(frame)

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    else:
        pass


if __name__ == '__main__':
    main()
