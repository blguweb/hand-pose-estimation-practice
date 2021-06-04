# 3D scene interactive application design based on gesture pose estimation
This is the content of the scientific research internship of my junior year. It is mainly based on the application design of gesture posture estimation and classification. It includes four scenarios: Perspective Transformation, gesture commands, free-hand movement, and left-and-right-handed games.

The basic network of this project is [SRHandNet](https://github.com/JiageWang/hand-pose-estimate) (gesture pose estimation) and [Classify-HandGesturePose](https://github.com/Prasad9/Classify-HandGesturePose) (gesture classification). Environment configuration can be obtained from there. 

# Instructions
## Preparation
1. Please go to their [README](https://github.com/lmb-freiburg/hand3d) pages to check the configuration of the environment and download [data](https://lmb.informatik.uni-freiburg.de/projects/hand3d/ColorHandPose3D_data_v3.zip),and unzip it into the projects root folder (This will create 3 folders: "data", "results" and "weights")
2. Download hand.pts in [SRHandNet](https://github.com/JiageWang/hand-pose-estimate) and put it into the projects root folder.
3. Download [videoDemo](https://github.com/JiageWang/hand-pose-estimate) into ./pose/
## Run the code

### Gesture command
Replace all the files in ./scenes/order in the projects root folder,especially
    *general.py in ./utils*
    *DeterminePositions.py in ./pose*
Run
``` python joint_control.py ./pose/pose_video/control.mp4 ```

### Perspective transformation
Replace all the files in ./scenes\viewTransformation in the projects root folder,and run
``` 
python UVideoRun.py ./pose/pose_video/variable_speed.mp4 
python UVideoRun.py ./pose/pose_video/direction_change.mp4
``` 

### Free-hand movement
Replace all the files in ./scenes\goWithTheFlow in the projects root folder,and run
``` 
python interation.py ./pose/pose_video/action.mp4
``` 

### Left-and-right-handed games
Replace all the files in ./scenes\games in the projects root folder,and run
``` 
python interationGame.py --pb-file=./Checkpoint/graph.pb 
``` 

