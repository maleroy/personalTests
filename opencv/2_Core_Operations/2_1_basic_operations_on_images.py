import numpy as np
import cv2
import matplotlib.pyplot as plt


BLUE = [255,0,0]


# Numpy indexing is normally used for selecting a region of array, 
# say first 5 rows and last 3 columns like that. 
# For individual pixel access, Numpy array methods, 
# array.item() and array.itemset() is considered to be better. 
# But it always returns a scalar. So if you want to access all B,G,R values, 
# you need to call array.item() separately for all.


def main():
    img1 = cv2.imread('../samples/hammerhead.jpg')

    print('Shape:', img1.shape)
    print('Size:', img1.size)
    print('DType:', img1.dtype)

    # need to change BLUE and RED Channels as it is going to be displayed in plt
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 

    replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant= cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)

    plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

    plt.show()


if __name__ == '__main__':
    main()
