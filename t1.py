#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    def dist(pt1, pt2):
        SUM = ((pt1 - pt2)**2).sum()
        D = np.sqrt(SUM)
        return D

    def getGoodPts(des1, des2, k=2):
        if len(des1)>len(des2):
            train = des1
            query = des2
        else:
            query = des1
            train = des2

        goodPts = []
        for tIdx, rowTrain in enumerate(train):
            row = []
            for qIdx, rowQuery in enumerate(query):
                D = dist(rowTrain, rowQuery)
                row.append((D, qIdx))
            row = sorted(row, key=lambda x:x[0])
            goodPts.append((row[:k], tIdx))
        return goodPts

    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1Gray,None)
    kp2, des2 = sift.detectAndCompute(img2Gray,None)
    
    matches = getGoodPts(des1, des2)

    goodPts = []
    for row in matches:
        m = row[0]
        if m[0][0] < 0.5*m[1][0]:
            goodPts.append(row)

    if len(des1)<len(des2):
        src = np.float32([ kp1[m[0][0][1]].pt for m in goodPts ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m[1]].pt for m in goodPts ]).reshape(-1,1,2)
    else:
        dst = np.float32([ kp2[m[0][0][1]].pt for m in goodPts ]).reshape(-1,1,2)
        src = np.float32([ kp1[m[1]].pt for m in goodPts ]).reshape(-1,1,2)
 
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    P = np.dot(H, np.array([0,0,1]))
    P = P/P[-1]

    if P[0]<0:
        xOff = int(np.ceil(abs(P[0])))
    else:
        xOff = 0
    
    yOff = 150

    H[0][-1] = P[0]+xOff
    H[1][-1] = P[1]+yOff

    warpedImg = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img1.shape[0] + img2.shape[0]))
    result = warpedImg.copy()
    result[yOff:img2.shape[0]+yOff, xOff:img2.shape[1]+xOff] = img2

    for r in range(300, 600):
        for c in range(350, 800):
            if warpedImg[r, c, 0] != 0 and warpedImg[r, c, 1] != 0 and warpedImg[r, c, 2] != 0:
                result[r, c, :] = warpedImg[r, c, :]
    
    cv2.imwrite(savepath, result)
    plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

