import cv2

th = 0
def change_thresh(val):
    th = cv2.getTrackbarPos('thresh', 'test')
    res = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow(title, res)

def nothing(a):
    pass
title = 'test'
img = cv2.imread('test.jpg', 0)
cv2.createTrackbar('thresh', 'test', 0, 255, nothing)
while(1):
    cv2.imshow(title, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

change_thresh(0)
cv2.waitKey()
