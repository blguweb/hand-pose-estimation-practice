from pose.utils.FingerCurled import FingerCurled
from pose.utils.FingerPosition import FingerPosition
from pose.utils.FingerDataFormation import FingerDataFormation

def determine_position(curled_positions, finger_positions, known_finger_poses, min_threshold):
    obtained_positions = {}
    
    for finger_pose in known_finger_poses:
        score_at = 0.0
        for known_curl, known_curl_confidence, given_curl in \
            zip(finger_pose.curl_position, finger_pose.curl_position_confidence, curled_positions):
                if len(known_curl) == 0:
                    if len(known_curl_confidence) == 1:
                        score_at += known_curl_confidence[0]
                        continue
                
                if given_curl in known_curl:
                    confidence_at = known_curl.index(given_curl)
                    score_at += known_curl_confidence[confidence_at]
                
        for known_position, known_position_confidence, given_position in \
            zip(finger_pose.finger_position, finger_pose.finger_position_confidence, finger_positions):
                if len(known_position) == 0:
                    if len(known_position_confidence) == 1:
                        score_at += known_position_confidence[0]
                        continue
                        
                if given_position in known_position:
                    confidence_at = known_position.index(given_position)
                    score_at += known_position_confidence[confidence_at]
        
        if score_at >= min_threshold:
            obtained_positions[finger_pose.position_name] = score_at
            
    return obtained_positions

def get_position_name_with_pose_id(pose_id, finger_poses):
    for finger_pose in finger_poses:
        if finger_pose.position_id == pose_id:
            return finger_pose.position_name
    return None


def create_known_finger_poses():
    known_finger_poses = []
    

    ####### 1 right
    right = FingerDataFormation()
    right.position_name = 'Right'
    right.curl_position = [
        [FingerCurled.NoCurl],   # Thumb
        [FingerCurled.NoCurl], # Index
        [FingerCurled.NoCurl], # Middle
        [FingerCurled.NoCurl], # Ring
        [FingerCurled.NoCurl]  # Little
    ]
    right.curl_position_confidence = [
        [1.0], # Thumb
        [1.0], # Index
        [1.0], # Middle
        [1.0], # Ring
        [1.0]  # Little
    ]
    right.finger_position = [
        [FingerPosition.DiagonalUpLeft,FingerPosition.VerticalUp], # Thumb
        [FingerPosition.HorizontalLeft, FingerPosition.DiagonalUpLeft], # Index
        [FingerPosition.HorizontalLeft, FingerPosition.DiagonalUpLeft], # Middle
        [FingerPosition.HorizontalLeft, FingerPosition.DiagonalUpLeft], # Ring
        [FingerPosition.HorizontalLeft, FingerPosition.DiagonalUpLeft] # Little
    ]
    right.finger_position_confidence = [
        [1.0, 0.5], # Thumb
        [1.0, 0.5], # Index
        [1.0, 0.5], # Middle
        [1.0, 0.5], # Ring
        [1.0, 0.5]  # Little
    ]
    right.position_id = 0
    known_finger_poses.append(right)



    ####### 2 observe
    observe = FingerDataFormation()
    observe.position_name = 'Observe'
    observe.curl_position = [
        [FingerCurled.NoCurl],   # Thumb
        [FingerCurled.HalfCurl], # Index
        [FingerCurled.HalfCurl], # Middle
        [FingerCurled.HalfCurl], # Ring
        [FingerCurled.HalfCurl]  # Little
    ]
    observe.curl_position_confidence = [
        [1.0], # Thumb
        [1.0], # Index
        [1.0], # Middle
        [1.0], # Ring
        [1.0]  # Little
    ]
    observe.finger_position = [
        [FingerPosition.DiagonalUpRight, FingerPosition.HorizontalRight], # Thumb
        [FingerPosition.DiagonalDownRight, FingerPosition.HorizontalRight], # Index
        [FingerPosition.DiagonalDownRight, FingerPosition.HorizontalRight], # Middle
        [FingerPosition.DiagonalDownRight, FingerPosition.HorizontalRight], # Ring
        [FingerPosition.DiagonalDownRight, FingerPosition.HorizontalRight] # Little
    ]
    observe.finger_position_confidence = [
        [1.0, 0.5], # Thumb
        [1.0, 0.5], # Index
        [1.0, 0.5], # Middle
        [1.0, 0.5], # Ring
        [1.0, 0.5]  # Little
    ]
    observe.position_id = 1
    known_finger_poses.append(observe)

    return known_finger_poses
