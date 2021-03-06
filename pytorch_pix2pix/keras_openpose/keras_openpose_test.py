import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

w = 256
h = 256
size = (256,256)

def process (input_image, params, model_params, pose_scale):

    oriImg = input_image  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    #for m in range(len(multiplier)):
    for m in range(1):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0
    
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10
    
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    #create a black use numpy
    poseFrame = np.zeros((h, w, 3), np.uint8)   
    #fill the image with black
    poseFrame.fill(1)

    # draw 18 keypoints
    keypoints = []
    for i in range(18):
        for j in range(len(all_peaks[i])):
            # loc = all_peaks[i][j][0:2]
            # print('x:', loc[0], ', y:', loc[1])
            # cv2.circle(poseFrame, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            keypoints.append(all_peaks[i][j][0:2])

    keypoints = normalize(keypoints, pose_scale)
    for i in range(len(keypoints) if len(keypoints) < 18 else 18):
            cv2.circle(poseFrame, keypoints[i], 4, colors[i], thickness=-1)

    # draw 17 parts of a body
    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_poseFrame = poseFrame.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            # normalize parts
            X = X * pose_scale[0]
            Y = Y * pose_scale[1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_poseFrame, polygon, colors[i])
            poseFrame = cv2.addWeighted(poseFrame, 0.4, cur_poseFrame, 0.6, 0)
    
    poseFrame, cur_radius = move_pose_center(input_image.shape, poseFrame)

    return poseFrame, cur_radius       


# normalize keypoints
def normalize(src_points, scale):

    normalized_points = []
    # 缩放
    mean_x = 0
    mean_y = 0
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
    print('>>> radius:', radius)
    radius = int(radius)
    # cv2.circle(poseFrame, center, radius, (255, 0, 0), 2)
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = np.float32([[1, 0, img_size[0]/2-x], [0, 1, img_size[1]/2-y]])
    dst = cv2.warpAffine(poseFrame, M, (img_size[1], img_size[0]))
    
    return dst, min(radius, img_size[0]/2, img_size[1]/2)

# 根据外接圆半径计算缩放比例
def getScale(pose_radius, model_radius=104.0):
    s = model_radius / pose_radius
    return (s, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../../images/pbug_man_450x420.avi', help='input video')
    parser.add_argument('--output', type=str, default='../../result/pose_out.avi', help='output pose video')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    keras_weights_file = args.model

    # load model
    tic = time.time()
    print('load model...')
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)
    print('* h5模型加载时间为：{:.2f}s.'.format(time.time() - tic))
    
    cap = cv2.VideoCapture(args.input if args.input else 0)
    # 视频总帧数
    frameNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # vedio writer
    # fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(* 'XVID')
    # 保存size必须和输出size设定为一致，否则无法写入保存文件
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    poseout = cv2.VideoWriter(args.output, fourcc, 20.0, size)

    start_time = time.time()
    print('start processing...')
    print('共计{}帧图像，预计耗时{:.2f}min.'.format(frameNum, frameNum * 1.75/60))
    j = 1
    scale = (0.7, 0.7)  # request x==y!! a bug.
    while(1):  
        ret, frame = cap.read()
        if not ret:
            break
        params, model_params = config_reader()
        # generate image with body parts
        poseFrame, pose_radius = process(frame, params, model_params, scale)
        scale = getScale(pose_radius)
        print('>>> scale = ', scale)
        # cur_pose + cur_frame 横向连接，图片作为pix2pix输入
        cur_pairs = np.concatenate([poseFrame, frame], axis=1)
        # write to pix2pix workdir
        cv2.imwrite('../pytorch_pix2pix/datasets/pbug_full/test/curPose.jpg', cur_pairs)
        # write in video
        poseout.write(poseFrame)
        end_time = time.time()
        cv2.imshow('frame', frame)
        cv2.imshow('poseFrame, scale', poseFrame)
        j += 1
        if j % 20 == 0:
            # 记录时间
            end_time = time.time()
            print('已处理{}/{}帧图像， 用时{:.4f}s， 平均每帧用时{:.4f}s'.format(j, int(frameNum), end_time - start_time, (end_time-start_time)/j))

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    end_time = time.time()
    print('{}张帧图像，处理完成！耗时{:.4f}s.'.format(j, end_time - start_time))
    
    cap.release()
    poseout.release()
    cv2.destroyAllWindows()    
