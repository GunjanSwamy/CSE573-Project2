# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json 


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
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

    img1 = imgs[0]
    for i in range(1, len(imgs)):
        img2 = imgs[i]

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
    
        yOff = int(np.ceil(abs(P[1])))
    
        H[0][-1] = P[0]+xOff
        H[1][-1] = P[1]+yOff


        result = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img1.shape[0] + img2.shape[0]))
        
        result[yOff:img2.shape[0]+yOff, xOff:img2.shape[1]+xOff] = img2
        
        img1 = result.copy()
    overlap_arr = []

    for img1 in imgs:
        temp = []
        for img2 in imgs:
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

            if len(goodPts)>4:
                temp.append(1)
            else:
                temp.append(0)

        overlap_arr.append(temp)

    cv2.imwrite(imgmark + "_stitched.png", result)
    plt.imshow(result)
    plt.show()

    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
