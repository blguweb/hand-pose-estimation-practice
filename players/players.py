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
# jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
jointPositions=[-0.5, -0.258, 0.31, -2.8, -0.30, 2.66, 2.32, 0.02, 0.02]
pandaEndEffectorIndex = 11  #8 11 
pandaNum = 7
rp = jointPositions

ll = [-2]*pandaNum
#upper limits for null space (todo: set them to proper range)
ul = [2]*pandaNum
#joint ranges for null space (todo: set them to proper range)
jr = [2]*pandaNum

class Games(object):

    def __init__(self, camera_attached=False):
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)
        
        self.lego_count = 20
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
        
        self.t = 0.

        self.legoindex = 0

        self.game_state = 0
        self.is_right = 0

    def reset(self):
        pass

    def load_robot(self):
        flags = pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        pybullet.setPhysicsEngineParameter(solverResidualThreshold=0)
        legos=[]
        # orn=[-0.707107, 0.0, 0.0, 0.707107] #p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        orn = [0.,0.,math.pi]
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.9450], globalScaling=1.5)
        # box = pybullet.loadURDF(BOX_URDF_PATH, [0.7, 0, 0], flags=flags, globalScaling=1.5)
        panda = pybullet.loadURDF(PANDA_URDF_PATH, [0,0,0], [0, 0, -1, 1], useFixedBase=True, flags=flags)
        panda2 = pybullet.loadURDF(PANDA_URDF_PATH, [1.4, 0, 0], [0, 0, 1, 1], useFixedBase=True, flags=flags)

        for i in range(self.lego_count):
            legox = random.uniform(-0.2, 0.2)
            legoy = random.uniform(-0.2, 0.2)
            offset = np.array([legox, legoy, 0.05])
            # print("offset",offset)
            legos.append(pybullet.loadURDF(LEGO_URDF_PATH,np.array([0.7, 0, 0])+offset,[0, 0, 0, 1], flags=flags))
            pybullet.changeVisualShape(legos[i],-1,rgbaColor=[random.random(),random.random(),random.random(),1])
        pybullet.setGravity(0,0,-9.8)
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
            text="The game will be completed with left and right hands. If there are two players, please use different hands to play.\
                    Game rules: There are a total of 20 legos on the table. After a text tip has appeared, each player gives the gesture indicates\
                     the number of legos they would take. A maximum of 5 can be taken at a time. The player who has taken all the Lego at\
                          the end wins. The game starts with the left hand by default.",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 0, 0],
            textSize=1,
            lifeTime=1
        )
        # print("test")
        self.game_state = 1
    
    def leftPlayer(self):
        pybullet.addUserDebugText(
            text="the left-hand player start to gesture how much legos you want to take !",
            textPosition=[0, -2, 1],
            textColorRGB=[1, 0, 0],
            textSize=1.2,
            lifeTime=4
        )
        
    
    def rightPlayer(self):
        pybullet.addUserDebugText(
            text="the right-hand player start to gesture how much legos you want to take !",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=4
        )
        
    
    def leftPlayerIndicatie(self):
        pybullet.addUserDebugText(
            text="Please check the player",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=4
        )
    
    def rightPlayerIndicatie(self):
        pybullet.addUserDebugText(
            text="Please check the player",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=4
        )
    
    def game_ending(self):
        if self.game_state == 3:
            pybullet.addUserDebugText(
            text="left-hand player wins!",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=4
            )
        elif self.game_state == 1:
            pybullet.addUserDebugText(
            text="right-hand player wins!",
            textPosition=[0, 0, 1],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=4
            )
    
    def leftPlayerOutput(self):
        state_time = 3.
        self.finger_target = 0.04
        self.gripper_height = 0.5
        ground_height = 0.0

        pybullet.stepSimulation()
        # ##慢慢的渲染
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)

        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        position = [position[0], position[1], self.gripper_height]
        print("POS",position)
        orn = pybullet.getQuaternionFromEuler([0.,math.pi,math.pi])
        #以末端作为姿态！
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)

        time.sleep(state_time)

        self.gripper_height = 0
        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        position = [position[0], position[1], self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        time.sleep(state_time)
        pybullet.stepSimulation()

        #target for fingers
        self.finger_target = 0.01
        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
            print("finger",self.finger_target)
        pybullet.stepSimulation()
        time.sleep(state_time)
        

        self.gripper_height = 0.5
        position = [position[0], position[1], self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        
        destination = -0.8 + self.legoindex * 0.01
        position = [0,destination, self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        self.gripper_height = 0.04
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        self.finger_target = 0.04
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        time.sleep(state_time)

        #复位
        # reset


    def rightPlayerOutput(self):
        state_time = 3.
        self.finger_target = 0.04
        self.gripper_height = 0.5
        ground_height = 0.0

        pybullet.stepSimulation()
        # ##慢慢的渲染
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)

        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        position = [position[0], position[1], self.gripper_height]
        print("POS",position)
        orn = pybullet.getQuaternionFromEuler([0.,math.pi,math.pi])
        #以末端作为姿态！
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)

        time.sleep(state_time)

        self.gripper_height = 0
        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        position = [position[0], position[1], self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        time.sleep(state_time)
        pybullet.stepSimulation()

        #target for fingers
        self.finger_target = 0.01
        position, o = pybullet.getBasePositionAndOrientation(self.legos[self.legoindex])
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
            print("finger",self.finger_target)
        pybullet.stepSimulation()
        time.sleep(state_time)
        

        self.gripper_height = 0.5
        position = [position[0], position[1], self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        
        destination = 0.8 + self.legoindex * 0.01
        position = [1.4,destination, self.gripper_height]
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        self.gripper_height = 0.04
        jointPoses = pybullet.calculateInverseKinematics(self.left_panda,pandaEndEffectorIndex, position, orn, ll, ul,
                jr, rp, maxNumIterations=20)
        for i in range(pandaNum):
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 30)
        time.sleep(state_time)

        self.finger_target = 0.04
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        for i in [9,10]:
            pybullet.setJointMotorControl2(self.left_panda, i, pybullet.POSITION_CONTROL,self.finger_target ,force= 10)
        time.sleep(state_time)





if __name__ == "__main__":
    ob = Games()
    # 
    ob.leftPlayer()
    while (1):
        ob.leftPlayerOutput()
        pybullet.setGravity(0, 0, -9.8)
        
        #p.stepSimulation(1./100.)
        time.sleep(1. / 240.)
