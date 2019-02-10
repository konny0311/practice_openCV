import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

if len(sys.argv) > 2:
    path = sys.argv[1]
    K = int(sys.argv[2])
    img = cv2.imread(path)
    Z = img.reshape((-1,3))
    u = np.unique(Z, axis = 0)
    print(len(u))
    print(u)
    # x = np.random.randint(25,50,(25,2))
    # y = np.random.randint(60,85,(25,2))
    # z = np.vstack((x,y))
    Z = np.float32(Z)
    #todo　ピクセル数が少ない色ほど正確に取りにくいっぽい
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(Z, K, None, criteria, 20, flags)
    u = np.unique(labels)
    print(u)
    print(len(u))
    centers = np.uint8(centers)
    print(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imwrite('pika_result.png', res2)
# A = z[labels.ravel()==0]
# B = z[labels.ravel()==1]
#
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1], c='r')
# plt.scatter(centers[:,0],centers[:,1], s=80, c='y', marker='s')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

# img = cv2.imread('digits.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
# x = np.array(cells)
#
# train = x[:,:50].reshape(-1, 400).astype(np.float32)
# test = x[:,50:100].reshape(-1, 400).astype(np.float32)
#
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()
#
# knn = cv2.ml.KNearest_create()
# knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.findNearest(test, 5)
#
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# acc = correct*100/result.size
# print(acc)


# trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# responses = np.random.randint(0,2,(25,1)).astype(np.float32)
#
# red = trainData[responses.ravel()==0]
# plt.scatter(red[:,0],red[:,1],80,'r','^')
#
# blue = trainData[responses.ravel()==1]
# plt.scatter(blue[:,0],blue[:,1],80,'b','s')
#
# # plt.show()
#
# new = np.random.randint(0,100,(1,2)).astype(np.float32)
# print(new)
# plt.scatter(new[:,0],new[:,1],80,'g','o')
#
# knn = cv2.ml.KNearest_create()
# knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# ret, results, neighbours, dist = knn.findNearest(new, 3)
#
# print('result:', results)
# print('neighbours:', neighbours)
# print('dist:', dist)
#
# plt.show()
