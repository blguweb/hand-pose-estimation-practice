import os
import math 
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import pybullet_data as pd
import ProcessFramesMoviePy as pf

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
# ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
OBJECT_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "random_urdfs/000/000.urdf")

class UR5Sim():
    
    def __init__(self, camera_attached=False):
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        
        self.roll = 0
        self.radius = 1
        self.speed = 1
        self.x = 0.8
        self.y = 0
        self.z = 1
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.Radius = 0.8
        self.step = 0.2
        self.angle = 0

        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info


    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        pybullet.setGravity(0,0,-10)
        objectUid = pybullet.loadURDF(OBJECT_URDF_PATH, basePosition=[0.9,0,0.1])
        return robot
    

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders(self):
        self.sliders = []# 滑块
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))


    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def rightProcess(self):
        # self.roll = math.atan(self.y / self.x)
        if self.roll <= 90 and self.roll >= -90:
            self.roll += self.speed
        if self.roll > 90:
            self.roll = 90
        print("roll",self.roll)
        # 根据起始位置不同需要调整相位，这里是[1, 0],注意单位转换
        cam_x = self.radius*np.cos(self.roll * np.pi / 180)
        cam_y = self.radius*np.sin(self.roll * np.pi / 180)
        # 设置相机位置
        pybullet.resetDebugVisualizerCamera(self.radius, self.roll+90, -30, [cam_x, cam_y, 1])
        pybullet.stepSimulation()

        self.x = self.Radius*np.cos(self.roll * np.pi / 180)
        self.y = self.Radius*np.sin(self.roll * np.pi / 180)
        self.rz = self.roll * np.pi / 180
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        # self.rx = 0
        # self.ry = 0
        # self.z = 1
        print("rz",self.x,self.y,self.rz)
        joint_angles = self.calculate_ik([self.x, self.y, self.z], [self.rx, self.ry, self.rz])
        self.set_joint_angles(joint_angles)

    def leftProcess(self):
        if self.roll <= 90 and self.roll >= -90:
            self.roll -= self.speed
        if self.roll < -90:
            self.roll = -90
        # print("roll",self.roll)
        # # 根据起始位置不同需要调整相位，这里是[1, 0],注意单位转换
        cam_x = self.radius*np.cos(self.roll * np.pi / 180)
        cam_y = self.radius*np.sin(self.roll * np.pi / 180)
        # 设置相机位置
        pybullet.resetDebugVisualizerCamera(self.radius, self.roll+90, -30, [cam_x, cam_y, 1])
        pybullet.stepSimulation()
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        self.x = self.Radius*np.cos(self.roll * np.pi / 180)
        self.y = self.Radius*np.sin(self.roll * np.pi / 180)
        self.rz = self.roll * np.pi / 180
        print("rz",self.x,self.y,self.rz)
        # self.rx = 0
        # self.ry = 0
        # self.z = 1
        joint_angles = self.calculate_ik([self.x, self.y, self.z], [self.rx, self.ry, self.rz])
        self.set_joint_angles(joint_angles)
    
    def upProcess(self):
        self.angle = math.atan(self.z / self.x)
        if self.angle <= 90 and self.angle >= -90:
            self.angle += self.speed
        if self.angle > 90:
            self.angle = 90
        print("angle",self.angle)
        # # 根据起始位置不同需要调整相位，这里是[1, 0],注意单位转换
        cam_x = self.radius*np.cos(self.angle * np.pi / 180)
        cam_z = self.radius*np.sin(self.angle * np.pi / 180)
        # 设置相机位置
        pybullet.resetDebugVisualizerCamera(self.radius, self.angle+90, -30, [cam_x, self.y, cam_z])
        pybullet.stepSimulation()

        self.x = self.Radius*np.cos(self.angle * np.pi / 180)
        self.z = self.Radius*np.sin(self.angle * np.pi / 180)
        self.ry = self.angle * np.pi / 180
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        # self.rx = 0
        # self.ry = 0
        joint_angles = self.calculate_ik([self.x, self.y, self.z], [self.rx, self.ry, self.rz])
        self.set_joint_angles(joint_angles)
    
    def downProcess(self):
        self.angle = math.atan(self.z / self.x)
        if self.angle <= 90 and self.angle >= -90:
            self.angle -= self.speed
        if self.angle < -90:
            self.angle = -90
        # print("angle",self.angle)
        # # 根据起始位置不同需要调整相位，这里是[1, 0],注意单位转换
        cam_x = self.radius*np.cos(self.angle * np.pi / 180)
        cam_z = self.radius*np.sin(self.angle * np.pi / 180)
        # 设置相机位置
        pybullet.resetDebugVisualizerCamera(self.radius, self.angle+90, -30, [cam_x, self.y, cam_z])
        pybullet.stepSimulation()

        self.x = self.Radius*np.cos(self.angle * np.pi / 180)
        self.z = self.Radius*np.sin(self.angle * np.pi / 180)
        self.ry = self.angle * np.pi / 180
        
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        # self.rx = 0
        # self.ry = 0
        joint_angles = self.calculate_ik([self.x, self.y, self.z], [self.rx, self.ry, self.rz])
        self.set_joint_angles(joint_angles)
    
    def circleProcess(self):
        self.x += self.step
        # self.y += self.step
        # self.z += self.step
        print("rw",self.x,self.y,self.z)
        for i in range(0, int(math.pi)):
            self.rx = float(i)
        self.rx = 0
        self.x -= self.step
        # self.y -= self.step
        # self.z -= self.step
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        joint_angles = self.calculate_ik([self.x, self.y, self.z], [self.rx, self.ry, self.rz])
        self.set_joint_angles(joint_angles)
        pybullet.stepSimulation()
        print("rz",self.x,self.y,self.z)
    
    def observeProcess(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        pybullet.setJointMotorControl2(self.ur5, 0, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.setJointMotorControl2(self.ur5, 1, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.setJointMotorControl2(self.ur5, 2, 
                        pybullet.POSITION_CONTROL,-math.pi/4.)
        pybullet.setJointMotorControl2(self.ur5, 3, 
                        pybullet.POSITION_CONTROL,math.pi/2.)
        pybullet.setJointMotorControl2(self.ur5, 4, 
                        pybullet.POSITION_CONTROL,-2*math.pi/2.)
        pybullet.setJointMotorControl2(self.ur5, 5, 
                        pybullet.POSITION_CONTROL,3*math.pi/2.)
        pybullet.setJointMotorControl2(self.ur5, 6, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.stepSimulation()
    
    def waitingProcess(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        pybullet.setJointMotorControl2(self.ur5, 0, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.setJointMotorControl2(self.ur5, 1, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.setJointMotorControl2(self.ur5, 2, 
                        pybullet.POSITION_CONTROL,-math.pi/2.)
        pybullet.setJointMotorControl2(self.ur5, 3, 
                        pybullet.POSITION_CONTROL,4*math.pi/5.)
        pybullet.setJointMotorControl2(self.ur5, 4, 
                        pybullet.POSITION_CONTROL,-5*math.pi/4.)
        pybullet.setJointMotorControl2(self.ur5, 5, 
                        pybullet.POSITION_CONTROL,3*math.pi/2.)
        pybullet.setJointMotorControl2(self.ur5, 6, 
                        pybullet.POSITION_CONTROL,0)
        pybullet.stepSimulation()
    
def demo_simulation(sim):
    """ Demo program showing how to use the sim """
    # sim = UR5Sim()
    # sim.add_gui_sliders()

    # x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
    joint_angles = sim.calculate_ik([sim.x, sim.y, sim.z], [sim.rx, sim.ry, sim.rz])
    sim.set_joint_angles(joint_angles)
    sim.check_collisions()
        

if __name__ == "__main__":
    
    demo_simulation()
