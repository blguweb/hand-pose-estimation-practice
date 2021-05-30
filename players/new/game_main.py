from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import os
import argparse
import operator
import csv
import cv2
import time
from moviepy.editor import VideoFileClip

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import *
from pose.utils.FingerPoseEstimate import FingerPoseEstimate
import players
import pybullet

# Variables to be used
# TODO: Check how to pass parameters through fl_image function. Remove global variables
image_tf = None
threshold = None
known_finger_poses = None
network_elements = None
output_txt_path = None
reqd_pose_name = None

fps=240.
timeStep = 1./fps

def parse_args():
    parser = argparse.ArgumentParser(description = 'Process frames in a video of a particular pose')
    parser.add_argument('video_path', help = 'Path of video', type = str)
    # This part needs improvement. Currently, pose_no is position_id present in FingerDataFormation.py
    # parser.add_argument('pose_no', help = 'Pose to classify at', type = int)
    parser.add_argument('--output-path', dest = 'output_path', type = str, default = None,
                        help = 'Path of folder where to store the text output')
    parser.add_argument('--thresh', dest = 'threshold', help = 'Threshold of confidence level(0-1)', default = 0.5,
                        type = float)
    args = parser.parse_args()
    return args

def prepare_paths(video_path, output_txt_path):
    video_path = os.path.abspath(video_path)

    if output_txt_path is None:
        output_txt_path = os.path.split(video_path)[0]
    else:
        output_txt_path = os.path.abspath(output_txt_path)
        if not os.path.exists(output_txt_path):
            os.mkdir(output_txt_path)

    file_name = os.path.basename(video_path).split('.')[0]
    output_video_path = os.path.join(output_txt_path, '{}_save.mp4'.format(file_name))
    # output_txt_path = os.path.join(output_txt_path, '{}.csv'.format(file_name))
    if not os.path.exists(output_txt_path):
        open(output_txt_path, 'w').close()
    return video_path,  output_video_path

def prepare_network():
    # network input
    image_tf = tf.placeholder(tf.float32, shape = (1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # Both left and right hands included
    evaluation = tf.placeholder_with_default(True, shape = ())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
        keypoints_scoremap_tf, keypoint_coord3d_tf= net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    return sess, image_tf, keypoint_coord3d_tf, scale_tf, center_tf, keypoints_scoremap_tf

def stringToNumber(label):
    num = 0
    if label == "One":
        num = 1
    elif label == "Two":
        num = 2
    elif label == "Three":
        num = 3
    elif label == "Four":
        num = 4
    return num

def leftPlayerOutput(sim):
    sim.state_t = 0
    sim.cur_state = 0
    for i in range (1250):
        # panda.bullet_client.submitProfileTiming("full_step")
        sim.step(sim.left_panda,0,-0.5,"left")
        pybullet.stepSimulation()
        # if createVideo:
        #     p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        # if not createVideo:
        time.sleep(sim.control_dt)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    sim.constraint_setting(sim.left_panda)

def rightPlayerOutput(sim):
    sim.state_t = 0
    sim.cur_state = 0
    for i in range (1250):# 3000
        # panda.bullet_client.submitProfileTiming("full_step")
        sim.step(sim.right_panda,1.0,0.5,"right")
        pybullet.stepSimulation()
        # if createVideo:
        #     p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        # if not createVideo:
        time.sleep(sim.control_dt)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    sim.constraint_setting(sim.right_panda)

class ProcessVideoFrame:
    def __init__(self,game):
        self.game = game
    def __call__(self, video_frame):
        video_frame = video_frame[:, :, :3]
        video_frame = scipy.misc.imresize(video_frame, (240, 320))
        image_v = np.expand_dims((video_frame.astype('float') / 255.0) - 0.5, 0)
 
        keypoint_coord3d_tf, scale_tf, center_tf, keypoints_scoremap_tf= network_elements
        keypoint_coord3d_v, scale_v, center_v, keypoints_scoremap_v= sess.run([keypoint_coord3d_tf,
            scale_tf, center_tf, keypoints_scoremap_tf], feed_dict = {image_tf: image_v})



        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        # post processing
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        plot_hand_2d(coord_hw, video_frame)

        score_label = process_keypoints(keypoint_coord3d_v)
        
        coord_four_hw = coord_hw[4, :]
        coord_twenty_hw = coord_hw[20, :]

        is_right_hand = coord_twenty_hw[1] - coord_four_hw[1] # u
        print("2",coord_four_hw)
        print("twenty",coord_twenty_hw)
        print("is right",is_right_hand)
        print("game_state",game.game_state)
        print("map",score_label)
        if game.lego_count <= 0:
            game.game_ending()
        else:
            if game.game_state == 1:
                # game.leftPlayerIndicatie()
                if score_label == "Fine":
                    game.is_right = is_right_hand
                    game.game_state = 2
            elif game.game_state == 2 and is_right_hand > 0 and score_label != None and score_label != "Fine":
                # game.leftPlayer()
                step = stringToNumber(score_label)
                print("step",step)
                print("size",type(step))
                if game.lego_count < step:
                    step = game.lego_count
                for j in range(step):
                    leftPlayerOutput(game)
                    game.legoindex = game.legoindex + 1
                    game.lego_count = game.lego_count -1
                    print("index",game.legoindex)
                game.game_state = 3
            elif game.game_state == 3:
                # game.rightPlayerIndicatie()
                if score_label == "Fine":
                    game.is_right = is_right_hand
                    game.game_state = 4
            elif game.game_state == 4 and is_right_hand < 0 and score_label != None and score_label != "Fine":
                # game.rightPlayer()
                step = stringToNumber(score_label)
                print("step",step)
                print("size",type(step))
                if game.lego_count < step:
                    step = game.lego_count
                for j in range(step):
                    rightPlayerOutput(game)
                    game.legoindex = game.legoindex + 1
                    game.lego_count = game.lego_count -1
                game.game_state = 1
        # if score_label is not None:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(video_frame, score_label, (10, 200), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        # print("flag",self.flag)
        # if score_label == 'Observe':
        #     self.ur5sim.observeProcess()

        # elif score_label == 'Right':
        #     self.ur5sim.waitingProcess()

        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        
        return video_frame


def process_keypoints(keypoint_coord3d_v):
    fingerPoseEstimate = FingerPoseEstimate(keypoint_coord3d_v)
    fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info = False)
    obtained_positions = determine_position(fingerPoseEstimate.finger_curled, 
                                        fingerPoseEstimate.finger_position, known_finger_poses,
                                        threshold)
    score_label = None
    if len(obtained_positions) > 0:
        max_pose_label = max(obtained_positions.items(), key=operator.itemgetter(1))[0]
        if obtained_positions[max_pose_label] >= threshold:
            score_label = max_pose_label
    
    return score_label

if __name__ == '__main__':
    args = parse_args()
    threshold = args.threshold * 10
    video_path, output_video_path = prepare_paths(args.video_path, args.output_path)
    known_finger_poses = create_known_finger_poses()
    
    sess, image_tf, keypoint_coord3d_tf, scale_tf, center_tf, keypoints_scoremap_tf = prepare_network()
    network_elements = [keypoint_coord3d_tf, scale_tf, center_tf, keypoints_scoremap_tf]


    game = players.PandaSimAuto()
    game.control_dt = timeStep
    game.start_game()
    time.sleep(1)
    # logId = pybullet.startStateLogging(pybullet.STATE_LOGGING_PROFILE_TIMINGS, "log.json")

    video_clip = VideoFileClip(video_path)
    white_clip = video_clip.fl_image(ProcessVideoFrame(game)) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_video_path, audio=False)
    # panda.bullet_client.stopStateLogging(logId)