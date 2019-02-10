import cv2
import numpy as np
img1 = cv2.imread('test.jpg')
img2 = cv2.imread('test.jpg')
# res = cv2.bitwise_not(img1, img2)
# white = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
# white.fill(255)
# res = cv2.absdiff(img1,white)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# res = cv2.inRange(gray,100,200)
cv2.imwrite('result.png', gray)
