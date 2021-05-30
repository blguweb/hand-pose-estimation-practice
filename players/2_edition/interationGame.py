import cv2
import torch
import numpy as np
import os
import csv
from collections import namedtuple
import pickle
import argparse
import tensorflow as tf
import scipy.misc
import time
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from pose.utils.FingerPoseEstimate import FingerPoseEstimate
import players
import pybullet

Point = namedtuple("Point", ["x", "y"])
Point.__add__ = lambda self, pair: Point(x=self.x + pair[0], y=self.y + pair[1])
Point.__sub__ = lambda self, pair: Point(x=self.x - pair[0], y=self.y - pair[1])
Point.__mul__ = lambda self, pair: Point(x=self.x * pair[0], y=self.y * pair[1])

Bbox = namedtuple("Bbox", ["x1", "y1", "x2", "y2"])

Ratio = namedtuple("Ratio", ["w", 'h'])
Ratio.__truediv__ = lambda self, pair: Ratio(w=self.w / pair[0], h=self.h / pair[1])
Ratio.__mul__ = lambda self, pair: Ratio(w=self.w * pair[0], h=self.h * pair[1])

Size = namedtuple("Size", ["w", 'h'])
Size.__truediv__ = lambda self, pair: Ratio(w=self.w / pair[0], h=self.h / pair[1])
Size.__mul__ = lambda self, pair: Size(w=self.w * pair[0], h=self.h * pair[1])
font = cv2.FONT_HERSHEY_SIMPLEX

threshold = None
known_finger_poses = None

fps=240.
timeStep = 1./fps
screen_x = 640

def parse_args():
    parser = argparse.ArgumentParser(description = 'Classify hand gestures from the set of images in folder')
    parser.add_argument('--pb-file', dest = 'pb_file', type = str, default = None,
                        help = 'Path where neural network graph is kept.')
    parser.add_argument('--thresh', dest = 'threshold', help = 'Threshold of confidence level(0-1)', default = 0.70,
	                    type = float)
    # If solving by SVM, give the path of svc pickle file.					
    args = parser.parse_args()
    return args

class HandCapture(object):
    CAP_SIZE = Size(640, 480)
    NET_SIZE = Size(256, 256)
    RATIO_CAP_TO_NET = NET_SIZE / CAP_SIZE
    RATIO_NET_DOWNSAMPLE = Ratio(4, 4)

    THRESHOLD = 0.4
    BLOCK_WIDTH = 2

    def __init__(self, model_path):
        if torch.cuda.is_available():
            model = torch.jit.load("hand.pts")
            model = model.cuda()
        else:
            model = torch.jit.load("hand.pts", map_location='cpu')
        self.model = model.eval()

    def to_tensor(self, img, bbox=None):
        input = img.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            input = input[y1:y2, x1:x2, :]
            ratio = self.NET_SIZE / Size(input.shape[1], input.shape[0])  # size/size=ratio
            M = np.array([[min(ratio), 0, 0], [0, min(ratio), 0]])
            input = cv2.warpAffine(input, M, self.NET_SIZE, borderMode=1, borderValue=128)
            # cv2.imshow('warp', input)
        else:
            ratio = self.RATIO_CAP_TO_NET
            input = cv2.resize(input, self.NET_SIZE)
        input = input.astype(float)
        input = input / 255 - 0.5
        tensor = torch.tensor(input, dtype=torch.float32)
        tensor = tensor.permute((2, 0, 1))
        tensor = tensor.unsqueeze(0)
        return tensor, ratio

    def forward(self, tensor):
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        featuremap = self.model(tensor)[3].cpu().data.numpy()  # (n,24,64,64)
        return featuremap

    def detectBbox(self, img):
        tensor, _ = self.to_tensor(img)
        featuremaps = self.forward(tensor)

        region_map = featuremaps[0, 21:, :, :]
        locations = self.nmsLocation(region_map[0])  # 找极大值点
        # 合成bbox
        bboxs = self.getBBox(region_map, locations)
        return bboxs

    def detectHand(self, img, bboxs):
        if not bboxs: return []
        tensors = []
        ratios = []
        for bbox in bboxs:
            tensor, ratio = self.to_tensor(img, bbox)
            tensors.append(tensor)
            ratios.append(ratio)
        input_tensors = torch.cat(tensors, 0)
        featuremaps = self.forward(input_tensors)

        keypoints = np.zeros((len(bboxs), 21,2))
        for i in range(len(bboxs)):
            for j in range(21):  # 21个关键点
                locations = self.nmsLocation(featuremaps[i, j, :, :])
                if locations:
                    point = locations[0][1] * self.RATIO_NET_DOWNSAMPLE
                    x = float(point.x / min(ratios[i]) + bboxs[i][0])
                    y = float(point.y / min(ratios[i]) + bboxs[i][1])
                    keypoints[i][j][0] = x
                    keypoints[i][j][1] = y
        return keypoints

    def getBBox(self, region_map, locations):
        """

        :param region_map: (3,64,464)
        :param locations: ((value, Point(x,y)),..)
        :param ratio: Ratio(w,h)
        :return:
        """
        bboxs = []
        for location in locations:
            point = location[1]  # (x, y)
            ratio_width = 0.  # 累加5x5内的ratio_width
            ratio_height = 0.  # 累加5x5内的ratio_height
            pixcount = 0  # 累加个数
            for m in range(max(point.y - 2, 0), min(point.y + 3, region_map.shape[1])):
                for n in range(max(point.x - 2, 0), min(point.x + 3, region_map.shape[2])):
                    ratio_width += region_map[1, m, n]
                    ratio_height += region_map[2, m, n]
                    pixcount += 1
            if pixcount > 0:
                ratio = Ratio(
                    min(max(ratio_width / pixcount, 0), 1),
                    min(max(ratio_height / pixcount, 0), 1)
                )  # 长宽相对于图像比例
                center = point * (self.RATIO_NET_DOWNSAMPLE / self.RATIO_CAP_TO_NET)  # (x,y)
                size = self.NET_SIZE * (ratio / self.RATIO_CAP_TO_NET)  # (w,h)
                x_min = int(max(center.x - size.w / 2, 0))
                y_min = int(max(center.y - size.h / 2, 0))
                x_max = int(min(center.x + size.w / 2, self.CAP_SIZE.w - 1))
                y_max = int(min(center.y + size.h / 2, self.CAP_SIZE.h - 1))
                bboxs.append(Bbox(x_min, y_min, x_max, y_max))  # (x, y, x, y)
        return bboxs

    def nmsLocation(self, featuremap):
        """
        :param featuremap:  特征图 64x64
        :return: locations ()
        """
        # set the local window size: 5*5
        locations = []
        blockwidth = self.BLOCK_WIDTH
        threshold = self.THRESHOLD
        for i in range(blockwidth, featuremap.shape[1] - blockwidth):  #
            for j in range(blockwidth, featuremap.shape[0] - blockwidth):
                value = featuremap[j][i]
                point = Point(i, j)  # (x,y)
                if value < threshold: continue
                localmaximum = True
                for m in range(min(i - blockwidth, 0), min(i + blockwidth, featuremap.shape[1] - 1) + 1):
                    for n in range(max(j - blockwidth, 0), min(j + blockwidth, featuremap.shape[0] - 1) + 1):
                        if featuremap[n][m] > value:
                            localmaximum = False
                            break
                    if not localmaximum: break
                if localmaximum:
                    locations.append((value, point))
        sorted(locations, key=lambda a: a[0], reverse=True)
        return locations

    @staticmethod
    def drawBbox(img, bboxs):
        if not bboxs: return
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

    def drawKeypoints(self, img, keypoints,length):
        # if not keypoints: return
        # for i in range(len(keypoints)):
        #     for keypoint in keypoints[i]:
        #         cv2.circle(img, keypoint, 2, (255, 0, 0))
        if keypoints is None: return
        for i in range(length):
            for j in range(21):
                # print("cire",keypoints[i][j][0],keypoints[i][j][1])
                x = int(keypoints[i][j][0])
                y = int(keypoints[i][j][1])
                cv2.circle(img, (x,y), 2, (255, 0, 0))

def predict_by_neural_network(keypoint_coord3d_v, known_finger_poses, pb_file, threshold):
    detection_graph = tf.Graph()
    score_label = 'Undefined'
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name = '')
            
        with tf.Session(graph = detection_graph) as sess:
            input_tensor = detection_graph.get_tensor_by_name('input:0')
            output_tensor = detection_graph.get_tensor_by_name('output:0')

            flat_keypoint = np.array([entry for sublist in keypoint_coord3d_v for entry in sublist])
            flat_keypoint = np.expand_dims(flat_keypoint, axis = 0)
            outputs = sess.run(output_tensor, feed_dict = {input_tensor: flat_keypoint})[0]

            max_index = np.argmax(outputs)
            score_index = max_index if outputs[max_index] >= threshold else -1
            score_label = 'Undefined' if score_index == -1 else get_position_name_with_pose_id(score_index, known_finger_poses) 
            print(outputs)
    return score_label

def stringToNumber(label):
    num = 0 # undefined
    if label == "One":
        num = 1
    elif label == "Two":
        num = 2
    elif label == "Three":
        num = 3
    elif label == "Four":
        num = 4
    elif label == "Five":
        num = 5
    return num

def leftPlayerOutput(sim):
    sim.state_t = 0
    sim.cur_state = 0
    sim.step(sim.left_panda,0,-0.5,"left",True)
    # for i in range (3000):
    #     # panda.bullet_client.submitProfileTiming("full_step")
    #     sim.step(sim.left_panda,0,-0.5,"left")
    #     pybullet.stepSimulation()
    #     # if createVideo:
    #     #     p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    #     # if not createVideo:
    #     time.sleep(sim.control_dt)
    sim.constraint_setting(sim.left_panda)

def rightPlayerOutput(sim):
    sim.state_t = 0
    sim.cur_state = 0
    sim.step(sim.right_panda,1.0,0.5,"right",True)
    # for i in range (3000):
    #     # panda.bullet_client.submitProfileTiming("full_step")
    #     sim.step(sim.right_panda,1.0,0.5,"right")
    #     pybullet.stepSimulation()
    #     # if createVideo:
    #     #     p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    #     # if not createVideo:
    #     time.sleep(sim.control_dt)
    sim.constraint_setting(sim.right_panda)

def gameState(game,keypoints,frame,pb_file,threshold):
    count = 0
    is_right_hand = None
    if game.game_state == 1:
        game.leftPlayerIndicatie()
    elif game.game_state == 2:
        game.rightPlayerIndicatie()
    if keypoints.size == 42:
        for i in range(21):
            key_node = keypoints[0,i,:]
            if key_node[0] == 0.0:
                count = count +1
        hand_keypoints = keypoints[0,:,:]
        print("coord",hand_keypoints)
        if count == 0:
            #left or right
            print("leftorright",hand_keypoints[0,0])
            if hand_keypoints[0,0] < screen_x/2:
                is_right_hand = 1
            else:
                is_right_hand = 0
            #state
            score_label = predict_by_neural_network(hand_keypoints, known_finger_poses,
                                                pb_file, threshold)
            step = stringToNumber(score_label)
            print("step",step)
            print("size",type(step))
            if game.lego_count < step:
                step = game.lego_count
            if game.game_state == 1 and is_right_hand == 0 and score_label != 'Undefined':
                cv2.putText(frame, score_label, (10, 200), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                game.leftPlayer(step)
                for j in range(step):
                    leftPlayerOutput(game)
                    game.legoindex = game.legoindex + 1
                    game.lego_count = game.lego_count -1
                    print("index",game.legoindex)
                game.game_state = 2
            if game.game_state == 2 and is_right_hand == 1 and score_label != 'Undefined':
                cv2.putText(frame, score_label, (10, 200), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                game.rightPlayer(step)
                for j in range(step):
                    rightPlayerOutput(game)
                    game.legoindex = game.legoindex + 1
                    game.lego_count = game.lego_count -1
                    print("index",game.legoindex)
                game.game_state = 1
    print("count",game.lego_count)
    if game.lego_count <= 0:
        game.game_ending()
        game.is_end = True


if __name__ == "__main__":
    hand = HandCapture("hand.pts")
    args = parse_args()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    known_finger_poses = create_known_finger_poses()
    thre = args.threshold
    pb_file = args.pb_file
    #
    game = players.PandaSimAuto()
    game.control_dt = timeStep
    game.start_game()
    time.sleep(1)
    while True:
        img = cap.read()[1]
        bboxs = hand.detectBbox(img)
        keypoints = hand.detectHand(img, bboxs)
        hand.drawBbox(img, bboxs)
        hand.drawKeypoints(img, keypoints,len(bboxs))
        keypoints = np.array(keypoints)
        # print("ke",type(keypoints),keypoints.shape,keypoints.size)
        if game.is_end == False:
            gameState(game,keypoints,img,pb_file,thre)
        
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break