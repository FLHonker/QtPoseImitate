# 归一化 'pose trick figure'
# 缩放 + 平移
# pbug_woman.radius = 103~105
import cv2
import numpy as np

# normalize keypoints
def normalize(src_points, scale):

    normalized_points = []
    # 缩放
    # mean_x = 0
    # mean_y = 0
    for i in range(len(src_points)):
        x = src_points[i][0] * scale[0]
        y = src_points[i][1] * scale[1]
        normalized_points.append((int(x), int(y)))
        # mean_x += x
        # mean_y += y

    # 平移到画布中央
    # mean_x = mean_x / len(normalized_points)
    # mean_y = mean_y / len(normalized_points)
    # move_x = img_size[1] / 2 - mean_x # shape[1] = width
    # move_y = img_size[0] / 2 - mean_y - 20    # shape[0] = height,dela配重
    # for i in range(len(normalized_points)):
    #     x = normalized_points[i][0] + move_x
    #     y = normalized_points[i][1] + move_y
    #     normalized_points[i] = (int(x), int(y))

    return normalized_points

# 平移pose到画布中央
def move_pose_center(img_size, poseFrame):
     
    # convert image to grayscale image
    gray_image = cv2.cvtColor(poseFrame, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    # cv2.circle(poseFrame, center, radius, (255, 0, 0), 2)
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = np.float32([[1, 0, img_size[0]/2-x], [0, 1, img_size[1]/2-y]])
    dst = cv2.warpAffine(poseFrame, M, (img_size[1], img_size[0]))
    
    return dst, radius 

# 根据外接圆半径计算缩放比例
def getScale(pose_radius, model_radius=104.0):
    s = model_radius / pose_radius
    return (s, s)