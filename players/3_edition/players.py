from __future__ import print_function, unicode_literals

import os
import numpy as np
import pybullet 
import pybullet_data as pd
import ProcessFramesMoviePy as pf
import time
import random
import math

## 规则 文字显示 左边先开始，文字显示的时候标志恢复正常 提取变成 0  ifcount = 0 则文字显示 游戏结束
# ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
# ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"
TABLE_URDF_PATH = os.path.join(pd.getDataPath(), "table/table.urdf")
BOX_URDF_PATH = os.path.join(pd.getDataPath(), "tray/traybox.urdf")
PANDA_URDF_PATH = os.path.join(pd.getDataPath(), "franka_panda/panda.urdf")
LEGO_URDF_PATH = os.path.join(pd.getDataPath(), "lego/lego.urdf")
BUILDING_URDF_PATH = os.path.join(pd.getDataPath(), "samurai.urdf")
# jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
jointPositions=[-0.5, -0.258, 0.31, -2.8, -0.30, 2.66, 2.32, 0.02, 0.02]
pandaEndEffectorIndex = 11  #8 11 
pandaNum = 7
rp = jointPositions
useNullSpace = 1
ikSolver = 0



ll = [-7]*pandaNum
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNum
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNum

class Games(object):

    def __init__(self):
        pybullet.connect(pybullet.GUI)
        # pybullet.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        pybullet.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=-math.pi/2, cameraPitch=-30, cameraTargetPosition=[0.7,0,0])
        
        self.lego_count = 10
        # self.end_effector_index = 7
        self.left_panda, self.right_panda, self.legos= self.load_robot()
        # self.left_panda, self.legos= self.load_robot()
        # self.num_joints = pybullet.getNumJoints(self.ur5)
        self.constraint_setting(self.left_panda)
        self.constraint_setting(self.right_panda)

        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        self.x = 0
        self.y = 0
        self.h = 0
        self.flag = False
        self.keypoints = np.zeros((1, 21,2))
        self.frame = None
        self.t = 0.

        self.legoindex = 0

        self.game_state = 0
        self.is_right = 0
        self.is_end = False

        self.state_t = 0
        self.cur_state = 0
        self.states=[0,3,5,4,6,3,7] # !!!
        for i in range(4):
            self.states.append(i+8)
        self.states.append(0)
        self.states.append(3)
        self.states.append(5)
        # print("states",self.states)
        self.state_durations=[0.5,0.5,0.5,1,0.5,0.5,1]
        for i in range(4):
            self.state_durations.append(1)
        self.state_durations.append(0.5)
        self.state_durations.append(0.5)
        self.state_durations.append(0.5)


    def reset(self):
        pass

    def load_robot(self):
        flags = pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        pybullet.setPhysicsEngineParameter(solverResidualThreshold=0)
        legos=[]
        # orn=[-0.707107, 0.0, 0.0, 0.707107] #p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        orn = [0.,0.,math.pi]
        # table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.9450], globalScaling=1.5)
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.4, 0, -0.94], globalScaling=1.5)
        # box = pybullet.loadURDF(BOX_URDF_PATH, [0.7, 0, 0], flags=flags)
        building = pybullet.loadURDF(BUILDING_URDF_PATH,[0,0,-0.945])
        panda = pybullet.loadURDF(PANDA_URDF_PATH, [0,0,0], [0, 0, -1, 1], useFixedBase=True, flags=flags)
        panda2 = pybullet.loadURDF(PANDA_URDF_PATH, [1, 0, 0], [0, 0, 1, 1], useFixedBase=True, flags=flags)

        legox = 0
        legoy = 0
        for i in range(self.lego_count):
            # legox = random.uniform(-0.2, 0.2)
            # while abs(legox-pre_x) < 0.01:
            #     pre_x = legox
            #     legox = random.uniform(-0.2, 0.2)
            # legoy = random.uniform(-0.2, 0.2)
            # while abs(legoy-pre_y) < 0.03:
            #     pre_y = legoy
            #     legoy = random.uniform(-0.2, 0.2)
            if i%4 == 0:
                legox += 0.06
                legoy = 0
            legoy += 0.06
            offset = np.array([legox, legoy, 0.05])
            print("offset",offset)
            legos.append(pybullet.loadURDF(LEGO_URDF_PATH,np.array([0.35, -0.12, 0])+offset,[0, 0, 0, 1], flags=flags,globalScaling=0.8))
            pybullet.changeVisualShape(legos[i],-1,rgbaColor=[random.random(),random.random(),random.random(),1])
        pybullet.setGravity(0,0,-9.8)
        pybullet.stepSimulation()
        time.sleep(1. / 2.)
        return panda, panda2, legos
        # return panda, legos
    
    def constraint_setting(self,robot):
        index = 0
        #create a constraint to keep the fingers centered
        c = pybullet.createConstraint(robot,
                        9,
                        robot,
                        10,
                        jointType=pybullet.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(pybullet.getNumJoints(robot)):
            pybullet.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
            info = pybullet.getJointInfo(robot, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == pybullet.JOINT_PRISMATIC):
                
                pybullet.resetJointState(robot, j, jointPositions[index]) 
                index=index+1
            if (jointType == pybullet.JOINT_REVOLUTE):
                pybullet.resetJointState(robot, j, jointPositions[index]) 
                index=index+1
    
    def start_game(self):
        #不能中文
        pybullet.addUserDebugText(
            text="The game will be completed with left and right hands. If there",
            textPosition=[-1, -1, 1.5],
            textColorRGB=[0, 0, 0],
            textSize=1,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text="are two players, please use different hands to play.",
            textPosition=[-1.05, -1, 1.3],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text="Game rules: There are a total of 20 legos on the table. After a text",
            textPosition=[-1.1, -1, 1.1],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text="tip has appeared, each player gives the gesture indicates the number",
            textPosition=[-1.15, -1, 0.9],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text="of legos they would take. A maximumof 5 can be taken at a time. The",
            textPosition=[-1.2, -1, 0.7],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text="player who has taken all the Lego at the end wins. The game starts",
            textPosition=[-1.25, -1, 0.5],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        pybullet.addUserDebugText(
            text=" with the left hand by default.",
            textPosition=[-1.3, -1, 0.3],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        # print("test")
        self.game_state = 1
    
    def leftPlayer(self,step):
        print("left",step)
        pybullet.addUserDebugText(
            text="the left-hand player chooses to grab "+str(step)+" legos!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        
    
    def rightPlayer(self,step):
        print("right",step)
        pybullet.addUserDebugText(
            text="the right-hand player chooses to grab "+str(step)+" legos!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=4
        )
        
    
    def leftPlayerIndicatie(self):
        pybullet.addUserDebugText(
            text="The left-handed player starts to indicate the number of grabs!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=1
        )
    
    def rightPlayerIndicatie(self):
        pybullet.addUserDebugText(
            text="The right-handed player starts to indicate the number of grabs ",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.0,
            lifeTime=1
        )
    
    def game_ending(self):
        print("test")
        if self.game_state == 2:
            pybullet.addUserDebugText(
            text="left-hand player wins!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.2
            )
        elif self.game_state == 1:
            pybullet.addUserDebugText(
            text="right-hand player wins!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.2
            )
    
    def legoCountIndication(self):
        pybullet.addUserDebugText(
            text="Please indicate the total number of legos!",
            textPosition=[-1, 0, 1],
            textColorRGB=[0, 0, 0],
            textSize=1.2,
            lifeTime=1
        )

    def update_state(self):
        
        f = True
        self.state_t += self.control_dt
        print("lenstate",len(self.state_durations),self.cur_state)
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state>=len(self.states):
                print("test")
                f = False
                return f
            else:
                self.state_t = 0
                self.state=self.states[self.cur_state]
        return f
    
    def step(self,panda,destination_x,destination_y,side,flag):
        while flag == True:
            if self.state==6:
                self.finger_target = 0.01
            if self.state==5:
                self.finger_target = 0.02
            flag = self.update_state()
            print("self.state=",self.state)
            # print("self.finger_target=",self.finger_target)
            alpha = 0.9 #0.99
            if self.state==1 or self.state==2 or self.state==3 or self.state==4 or self.state >=7:
                #gripper_height = 0.034
                if self.state == 1 or self.state == 4 :
                    self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.01
                if self.state == 2 or self.state == 3 :
                    self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.25

                
                t = self.t
                self.t += self.control_dt
                # pos = [0.6 + 0.1 * math.cos(1.5 * t), 0.2 * math.sin(1.5 * t) ,self.gripper_height]
                if self.state == 3 or self.state== 4:
                    pos, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
                    pos = [pos[0], pos[1],self.gripper_height]
                    self.prev_pos = pos
                    self.flag = True
                if self.state >= 7:
                    self.gripper_height = 0.25
                    divide = 5
                    destination_h = 0.5
                    if self.flag == True:
                        self.y = (self.prev_pos[1] - destination_y) / divide * self.control_dt
                        self.x = (self.prev_pos[0] - destination_x) / divide * self.control_dt
                        self.flag = False
                        self.h = (destination_h - self.gripper_height)/ divide * self.control_dt
                    # self.gripper_height = self.gripper_height + self.h
                    pos = [self.prev_pos[0] - self.x,self.prev_pos[1] - self.y, self.gripper_height]
                    self.prev_pos = pos
                    print("h",self.h)
                    print("pre", self.prev_pos)
                    print("pos" ,pos)
                    # pos = des
                    # pos = [self.prev_pos[0]-(self.prev_pos[0] - 0) / 2, self.prev_pos[1]- (self.prev_pos[0] + 0.5) / 2, self.prev_pos[2]]
                    # pos = self.prev_pos
                    # diffY = pos[1] - self.offset[0]
                    # # diffZ = pos[2] - (self.offset[2]-0.6)
                    # diffX = pos[0] - (self.offset[1]+0.6)
                    # self.prev_pos = [self.prev_pos[0]-diffX*0.1 , self.prev_pos[1]- diffY*0.1, self.prev_pos[2]]
                    # 
                    # if self.state == 8:
                    #   pos = [0, -0.5,self.gripper_height]
                    #   print("pos",pos)
                    # orn = pybullet.getQuaternionFromEuler([math.pi/2.,0.,0.])
                if side == "left":
                    orn = pybullet.getQuaternionFromEuler([0.,math.pi,math.pi])
                else:
                    # orn = pybullet.getQuaternionFromEuler([0,math.pi,-math.pi])
                    orn = pybullet.getQuaternionFromEuler([0,math.pi,0])
                jointPoses = pybullet.calculateInverseKinematics(panda,pandaEndEffectorIndex, pos, orn, ll, ul,
                    jr, rp, maxNumIterations=20)
                for i in range(pandaNum):
                    pybullet.setJointMotorControl2(panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
                #target for fingers
            for i in [9,10]:
                pybullet.setJointMotorControl2(panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
            print("height",self.gripper_height)
            pybullet.stepSimulation()
    

    

# class PandaSimAuto(Games):
#     def __init__(self):
        
#         # print("dur",self.state_durations)

    


if __name__ == "__main__":
    ob = Games()