# 归一化 'pose trick figure'
# 缩放 + 平移
import glob as gb
import cv2 as cv
from keras_openpose.config_reader import config_reader

# 根据训练集人物大小计算缩放比例
def getScale():
    img_path = gb.glob('./datasets/pbug_full/train/*.jpg')  #读取train文件夹下所有图像文件的名字
    if not img_path:
        return None
    image = cv.imread(img_path[0])
    image = image[:][image.width/2:]
    params, model_params = config_reader()
    keypoints = get_src_keypoints(image, params, model_params)
    

def normalize(src_points, scale):
    normalized_points = []
    # 缩放
    top_p = (999, 999)
    bottom_p1 = (0, 0)
    bottom_p2 = (0, 0)
    for i in range(len(src_points)):
        normalized_points.append((src_points[0][i] * scale[0], src_points[i][1] * scale[1]))
        if normalized_points[i][1] < top_p[1]:
            top_p = normalized_points[i]
        if normalized_points[i][1] > bottom_p1[1]:
            bottom_p1 = normalized_points[i]
        elif normalized_points[i][1] > bottom_p2[1]:
            bottom_p2 = normalized_points[i]

    # 平移到画布中央
    mid_x = (bottom_p1[0] + bottom_p2[0]) / 2
    mid_y = (bottom_p1[1] + top_p[1]) / 2
    move_x = 256 / 2 - mid_x
    move_y = 256 / 2 - mid_y
    for i in range(len(normalized_points)):
        normalized_points[i][0] += move_x
        normalized_points[i][1] += move_y

    return normalized_points


def get_src_keypoints(input_image, params, model_params):

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
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
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

    # 18 keypoints
    keypoints = []
    for i in range(18):
        for j in range(len(all_peaks[i])):
            keypoints.append(all_peaks[i][j][0:2])
                
    return keypoints
  