import cv2

img = cv2.imread('test.png', 1)
bordered = cv2.copyMakeBorder(img, 10,10,10,10, cv2.BORDER_CONSTANT, value=[255,0,0])
cv2.imwrite('result.png', bordered)
