import pandas as pd
import cv2
import mediapipe as mp
import numpy as np

#Inicializador media pipe#
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5
                                  )

pose = mp_pose.Pose(
                    static_image_mode=False,  # Para video, se debe establecer en False
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                    )

# Funciones
###Face####
# Function to calculate distance between landmarks
def distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to estimate head pose angles
def get_head_pose(face_landmarks, image_width, image_height):
    # 3D model points (from MediaPipe documentation)
    model_points = np.array([
                             (0.0, 0.0, 0.0),             # Nose tip
                             (0.0, -330.0, -65.0),        # Chin
                             (-225.0, 170.0, -135.0),     # Left eye left corner
                             (225.0, 170.0, -135.0),      # Right eye right corner
                             (-150.0, -150.0, -125.0),    # Left Mouth corner
                             (150.0, -150.0, -125.0)      # Right mouth corner
                             ])
        
                             # 2D image points (from detected landmarks)
    image_points = np.array([
                            (face_landmarks.landmark[1].x * image_width, face_landmarks.landmark[1].y * image_height),
                            (face_landmarks.landmark[152].x * image_width, face_landmarks.landmark[152].y * image_height),
                            (face_landmarks.landmark[33].x * image_width, face_landmarks.landmark[33].y * image_height),
                            (face_landmarks.landmark[263].x * image_width, face_landmarks.landmark[263].y * image_height),
                            (face_landmarks.landmark[61].x * image_width, face_landmarks.landmark[61].y * image_height),
                            (face_landmarks.landmark[291].x * image_width, face_landmarks.landmark[291].y * image_height)
                            ], dtype="double")
                            
    # Camera internals
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
     
    # Solve for head pose
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Get rotation angles (in degrees)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
                             
    return pitch, yaw, roll

###Body###

def calculate_bone_orientation(landmark1, landmark2):
    """Calcula la orientación de un hueso en los planos ZX, XY, YZ."""
    v = np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])
    
    # Orientación en el plano ZX
    zx_angle = np.degrees(np.arctan2(v[2], v[0]))
    
    # Orientación en el plano XY
    xy_angle = np.degrees(np.arctan2(v[1], v[0]))
    
    # Orientación en el plano YZ
    yz_angle = np.degrees(np.arctan2(v[2], v[1]))
    
    return zx_angle, xy_angle, yz_angle

def calculate_angle(landmark1, landmark2, landmark3):
    """Calcula el ángulo entre tres puntos de referencia."""
    v1 = np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])
    v2 = np.array([landmark3.x - landmark2.x, landmark3.y - landmark2.y, landmark3.z - landmark2.z])
    
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def calculate_bone_orientation(landmark1, landmark2):
    """Calcula la orientación de un hueso en los planos ZX, XY, YZ."""
    v = np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])
    
    # Orientación en el plano ZX
    zx_angle = np.degrees(np.arctan2(v[2], v[0]))
    
    # Orientación en el plano XY
    xy_angle = np.degrees(np.arctan2(v[1], v[0]))
    
    # Orientación en el plano YZ
    yz_angle = np.degrees(np.arctan2(v[2], v[1]))
    
    return zx_angle, xy_angle, yz_angle

def calculate_midpoint(landmark1, landmark2):
    return mp.solutions.pose.PoseLandmark(
                                          x=(landmark1.x + landmark2.x) / 2,
                                          y=(landmark1.y + landmark2.y) / 2,
                                          z=(landmark1.z + landmark2.z) / 2,
                                          )

def calculate_midpoint(landmark1, landmark2):
    return mp.solutions.pose.PoseLandmark(
                                          x=(landmark1.x + landmark2.x) / 2,
                                          y=(landmark1.y + landmark2.y) / 2,
                                          z=(landmark1.z + landmark2.z) / 2,
                                          )
def calculate_stdv(landmark1, landmark2):
    """Calcula la desviación estándar de dos landmarks."""
    x_values = [landmark1.x, landmark2.x]
    y_values = [landmark1.y, landmark2.y]
    z_values = [landmark1.z, landmark2.z]
    
    x_stdv = np.std(x_values)
    y_stdv = np.std(y_values)
    z_stdv = np.std(z_values)
    
    return mp.solutions.pose.PoseLandmark(x=x_stdv, y=y_stdv, z=z_stdv)

###Extractores de features###

def extract_facial_features(face_landmarks, image_width, image_height):
    """Extrae las features faciales de los landmarks."""
    
    if face_landmarks:
        pitch, yaw, roll = get_head_pose(face_landmarks, image_width, image_height)
        mouth_open = distance(face_landmarks.landmark[13], face_landmarks.landmark[14])
        left_eye_closed = distance(face_landmarks.landmark[159], face_landmarks.landmark[145]) < 0.02
        right_eye_closed = distance(face_landmarks.landmark[386], face_landmarks.landmark[374]) < 0.02
        left_eyebrow_lowered = face_landmarks.landmark[105].y > face_landmarks.landmark[107].y
        left_eyebrow_raised = face_landmarks.landmark[105].y < face_landmarks.landmark[107].y
        right_eyebrow_lowered = face_landmarks.landmark[334].y > face_landmarks.landmark[336].y
        right_eyebrow_raised = face_landmarks.landmark[334].y < face_landmarks.landmark[336].y
        left_eye_center_x = (face_landmarks.landmark[159].x + face_landmarks.landmark[145].x) / 2
        right_eye_center_x = (face_landmarks.landmark[386].x + face_landmarks.landmark[374].x) / 2
        gaze_direction_forward = abs(left_eye_center_x - right_eye_center_x) < 0.05
        gaze_direction_left = left_eye_center_x < right_eye_center_x
        gaze_direction_right = left_eye_center_x > right_eye_center_x
        au01_inner_brow_raiser = face_landmarks.landmark[66].y - face_landmarks.landmark[27].y
        au02_outer_brow_raiser = face_landmarks.landmark[107].y - face_landmarks.landmark[52].y
        au04_brow_lowerer = face_landmarks.landmark[52].y - face_landmarks.landmark[107].y
        au05_upper_lid_raiser = distance(face_landmarks.landmark[159], face_landmarks.landmark[145])
        au06_cheek_raiser = distance(face_landmarks.landmark[127], face_landmarks.landmark[234])
        au07_lid_tightener = distance(face_landmarks.landmark[33], face_landmarks.landmark[133])
        au09_nose_wrinkler = distance(face_landmarks.landmark[1], face_landmarks.landmark[5])
        au10_upper_lip_raiser = face_landmarks.landmark[13].y - face_landmarks.landmark[0].y
        au12_lip_corner_puller = distance(face_landmarks.landmark[78], face_landmarks.landmark[308])
        au14_dimpler = distance(face_landmarks.landmark[61], face_landmarks.landmark[291])
        au15_lip_corner_depressor = face_landmarks.landmark[0].y - face_landmarks.landmark[14].y
        au17_chin_raiser = distance(face_landmarks.landmark[152], face_landmarks.landmark[176])
        au20_lip_stretcher = distance(face_landmarks.landmark[57], face_landmarks.landmark[287])
        au23_lip_tightener = distance(face_landmarks.landmark[13], face_landmarks.landmark[14])
        au24_lip_pressor = distance(face_landmarks.landmark[0], face_landmarks.landmark[17])
        au25_lips_part = distance(face_landmarks.landmark[13], face_landmarks.landmark[14])
        au26_jaw_drop = distance(face_landmarks.landmark[13], face_landmarks.landmark[152])
        au27_mouth_stretch = distance(face_landmarks.landmark[61], face_landmarks.landmark[291])
        au43_eyes_closed = left_eye_closed and right_eye_closed
        features = {
              'SyHeadOrientation': pitch,
              'SxHeadOrientation': yaw,
              'SzHeadOrientation': roll,
              'SmouthOpen': mouth_open,
              'SleftEyeClosed': left_eye_closed,
              'SrightEyeClosed': right_eye_closed,
              'SleftEyebrowLowered': left_eyebrow_lowered,
              'SleftEyebrowRaised': left_eyebrow_raised,
              'SrightEyebrowLowered': right_eyebrow_lowered,
              'SrightEyebrowRaised': right_eyebrow_raised,
              'SgazeDirectionForward': gaze_direction_forward,
              'SgazeDirectionLeft': gaze_direction_left,
              'SgazeDirectionRight': gaze_direction_right,
              'SAu01_InnerBrowRaiser': au01_inner_brow_raiser,
              'SAu02_OuterBrowRaiser': au02_outer_brow_raiser,
              'SAu04_BrowLowerer': au04_brow_lowerer,
              'SAu05_UpperLidRaiser': au05_upper_lid_raiser,
              'SAu06_CheekRaiser': au06_cheek_raiser,
              'SAu07_LidTightener': au07_lid_tightener,
              'SAu09_NoseWrinkler': au09_nose_wrinkler,
              'SAu10_UpperLipRaiser': au10_upper_lip_raiser,
              'SAu12_LipCornerPuller': au12_lip_corner_puller,
              'SAu14_Dimpler': au14_dimpler,
              'SAu15_LipCornerDepressor': au15_lip_corner_depressor,
              'SAu17_ChinRaiser': au17_chin_raiser,
              'SAu20_LipStretcher': au20_lip_stretcher,
              'SAu23_LipTightener': au23_lip_tightener,
              'SAu24_LipPressor': au24_lip_pressor,
              'SAu25_LipsPart': au25_lips_part,
              'SAu26_JawDrop': au26_jaw_drop,
              'SAu27_MouthStretch': au27_mouth_stretch,
              'SAu43_EyesClosed': au43_eyes_closed
          }

    return features


def extract_body_features(landmarks):
    """Extrae las características del cuerpo a partir de los puntos de referencia."""
    features = []
    
    # Distancia
    features.append(landmarks[mp_pose.PoseLandmark.NOSE].z)  # avgDepth (aproximación)
    
    # Ángulos articulares
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]))  # leftShoulderAngle
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]))  # rightShoulderAngle
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]))  # leanAngle (aproximación)
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])) # HipCenter_Spine-Spine_ShoulderCenter
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.NOSE])) # Spine_ShoulderCenter-ShoulderCenter_Head
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])) # Spine_ShoulderCenter-ShoulderCenter_ShoulderLeft
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])) # Spine_ShoulderCenter-ShoulderCenter_ShoulderRight
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_WRIST])) # ShoulderCenter_ShoulderLeft-ShoulderLeft_ElbowLeft
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.LEFT_PINKY])) # ShoulderLeft_ElbowLeft-ElbowLeft_WristLeft
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])) # ShoulderCenter_ShoulderRight-ShoulderRight_ElbowRight
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.LEFT_PINKY], landmarks[mp_pose.PoseLandmark.LEFT_INDEX])) # ElbowLeft_WristLeft-WristLeft_HandLeft (aproximación)
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])) # ShoulderCenter_ShoulderRight-ShoulderRight_ElbowRight
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_PINKY])) # ShoulderRight_ElbowRight-ElbowRight_WristRight
    features.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_PINKY], landmarks[mp_pose.PoseLandmark.RIGHT_INDEX])) # ElbowRight_WristRight-WristRight_HandRight (aproximación)
    
    
    # Orientaciones óseas
    bones = [
             (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER),  # HipCenter_Spine
             (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # Spine_ShoulderCenter
             (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE),  # ShoulderCenter_Head
             (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),  # ShoulderCenter_ShoulderLeft
             (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # ShoulderCenter_ShoulderRight
             (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),  # ShoulderLeft_ElbowLeft
             (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),  # ElbowLeft_WristLeft
             (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),  # WristLeft_HandLeft
             (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),  # ShoulderRight_ElbowRight
             (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),  # ElbowRight_WristRight
             (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY),  # WristRight_HandRight
             ]
    for landmark1_index, landmark2_index in bones:
         zx_angle, xy_angle, yz_angle = calculate_bone_orientation(landmarks[landmark1_index], landmarks[landmark2_index])
         features.extend([zx_angle, xy_angle, yz_angle])

    return features

###Extrator total de features###

def extract_all_features(image, face_mesh, pose):
    """Extrae las features faciales y corporales de una imagen."""
    
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen con los extractores de features
    results_face_mesh = face_mesh.process(image_rgb)
    results_pose = pose.process(image_rgb)
    
    # Inicializar las features
    all_features = {}
    
    if results_face_mesh.multi_face_landmarks:
        face_landmarks = results_face_mesh.multi_face_landmarks[0]
        facial_features = extract_facial_features(face_landmarks, image.shape[1], image.shape[0])
        all_features.update(facial_features)
    
    # Extraer features corporales
    if results_pose.pose_landmarks:
        body_landmarks = results_pose.pose_landmarks.landmark
        body_posture_features = extract_body_features(body_landmarks)  # Usar la función extract_features del segundo extractor
        # Agregar las features corporales al diccionario all_features
        all_features.update({'body_' + k: v for k, v in zip(['avgDepth', 'leftShoulderAngle', 'rightShoulderAngle', 'leanAngle', 'HipCenter_SpineSpine_ShoulderCenter', 'Spine_ShoulderCenterShoulderCenter_Head', 'Spine_ShoulderCenterShoulderCenter_ShoulderLeft', 'Spine_ShoulderCenterShoulderCenter_ShoulderRight', 'ShoulderCenter_ShoulderLeftShoulderLeft_ElbowLeft', 'ShoulderLeft_ElbowLeftElbowLeft_WristLeft', 'ShoulderCenter_ShoulderRightShoulderRight_ElbowRight', 'ElbowLeft_WristLeftWristLeft_HandLeft', 'ShoulderCenter_ShoulderRightShoulderRight_ElbowRight', 'ShoulderRight_ElbowRightElbowRight_WristRight', 'ElbowRight_WristRightWristRight_HandRight', 'HipCenter_SpinePlaneZXAxisX', 'HipCenter_SpinePlaneXYAxisY', 'HipCenter_SpinePlaneYZAxisZ', 'Spine_ShoulderCenterPlaneZXAxisX', 'Spine_ShoulderCenterPlaneXYAxisY', 'Spine_ShoulderCenterPlaneYZAxisZ', 'ShoulderCenter_HeadPlaneZXAxisX', 'ShoulderCenter_HeadPlaneXYAxisY', 'ShoulderCenter_HeadPlaneYZAxisZ', 'ShoulderCenter_ShoulderLeftPlaneZXAxisX', 'ShoulderCenter_ShoulderLeftPlaneXYAxisY', 'ShoulderCenter_ShoulderLeftPlaneYZAxisZavg', 'ShoulderCenter_ShoulderRightPlaneZXAxisXavg', 'ShoulderCenter_ShoulderRightPlaneXYAxisY', 'ShoulderCenter_ShoulderRightPlaneYZAxisZ', 'ShoulderLeft_ElbowLeftPlaneZXAxisX', 'ShoulderLeft_ElbowLeftPlaneXYAxisY', 'ShoulderLeft_ElbowLeftPlaneYZAxisZ', 'ElbowLeft_WristLeftPlaneZXAxisX', 'ElbowLeft_WristLeftPlaneXYAxisY', 'ElbowLeft_WristLeftPlaneYZAxisZ', 'WristLeft_HandLeftPlaneZXAxisX', 'WristLeft_HandLeftPlaneXYAxisY', 'WristLeft_HandLeftPlaneYZAxisZ', 'ShoulderRight_ElbowRightPlaneZXAxisX', 'ShoulderRight_ElbowRightPlaneXYAxisY', 'ShoulderRight_ElbowRightPlaneYZAxisZ', 'ElbowRight_WristRightPlaneZXAxisX', 'ElbowRight_WristRightPlaneXYAxisY', 'ElbowRight_WristRightPlaneYZAxisZ', 'WristRight_HandRightPlaneZXAxisX', 'WristRight_HandRightPlaneXYAxisY', 'WristRight_HandRightKinectZAxis'], body_posture_features)})  # Agregar prefijo 'body_' a las claves
        all_features = {k: (1 if v is True else 0) if isinstance(v, bool) else v for k, v in all_features.items()}
    return all_features

def extract_all_features_local(image):
    """
    Extrae las features faciales y corporales de una imagen.
    Esta versión crea instancias locales de face_mesh y pose para evitar problemas de serialización.
    """
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inicializar las instancias de Mediapipe localmente
    with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, \
         mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        results_face_mesh = face_mesh.process(image_rgb)
        results_pose = pose.process(image_rgb)
        
        # Inicializar las features
        all_features = {}
        
        if results_face_mesh.multi_face_landmarks:
            face_landmarks = results_face_mesh.multi_face_landmarks[0]
            facial_features = extract_facial_features(face_landmarks, image.shape[1], image.shape[0])
            all_features.update(facial_features)
        
        if results_pose.pose_landmarks:
            body_landmarks = results_pose.pose_landmarks.landmark
            body_posture_features = extract_body_features(body_landmarks)
            keys = [
                'avgDepth', 'leftShoulderAngle', 'rightShoulderAngle', 'leanAngle', 
                'HipCenter_SpineSpine_ShoulderCenter', 'Spine_ShoulderCenterShoulderCenter_Head', 
                'Spine_ShoulderCenterShoulderCenter_ShoulderLeft', 'Spine_ShoulderCenterShoulderCenter_ShoulderRight', 
                'ShoulderCenter_ShoulderLeftShoulderLeft_ElbowLeft', 'ShoulderLeft_ElbowLeftElbowLeft_WristLeft', 
                'ShoulderCenter_ShoulderRightShoulderRight_ElbowRight', 'ElbowLeft_WristLeftWristLeft_HandLeft', 
                'ShoulderCenter_ShoulderRightShoulderRight_ElbowRight', 'ShoulderRight_ElbowRightElbowRight_WristRight', 
                'ElbowRight_WristRightWristRight_HandRight', 'HipCenter_SpinePlaneZXAxisX', 
                'HipCenter_SpinePlaneXYAxisY', 'HipCenter_SpinePlaneYZAxisZ', 'Spine_ShoulderCenterPlaneZXAxisX', 
                'Spine_ShoulderCenterPlaneXYAxisY', 'Spine_ShoulderCenterPlaneYZAxisZ', 
                'ShoulderCenter_HeadPlaneZXAxisX', 'ShoulderCenter_HeadPlaneXYAxisY', 
                'ShoulderCenter_HeadPlaneYZAxisZ', 'ShoulderCenter_ShoulderLeftPlaneZXAxisX', 
                'ShoulderCenter_ShoulderLeftPlaneXYAxisY', 'ShoulderCenter_ShoulderLeftPlaneYZAxisZavg', 
                'ShoulderCenter_ShoulderRightPlaneZXAxisXavg', 'ShoulderCenter_ShoulderRightPlaneXYAxisY', 
                'ShoulderCenter_ShoulderRightPlaneYZAxisZ', 'ShoulderLeft_ElbowLeftPlaneZXAxisX', 
                'ShoulderLeft_ElbowLeftPlaneXYAxisY', 'ShoulderLeft_ElbowLeftPlaneYZAxisZ', 
                'ElbowLeft_WristLeftPlaneZXAxisX', 'ElbowLeft_WristLeftPlaneXYAxisY', 
                'ElbowLeft_WristLeftPlaneYZAxisZ', 'WristLeft_HandLeftPlaneZXAxisX', 
                'WristLeft_HandLeftPlaneXYAxisY', 'WristLeft_HandLeftPlaneYZAxisZ', 
                'ShoulderRight_ElbowRightPlaneZXAxisX', 'ShoulderRight_ElbowRightPlaneXYAxisY', 
                'ShoulderRight_ElbowRightPlaneYZAxisZ', 'ElbowRight_WristRightPlaneZXAxisX', 
                'ElbowRight_WristRightPlaneXYAxisY', 'ElbowRight_WristRightPlaneYZAxisZ', 
                'WristRight_HandRightPlaneZXAxisX', 'WristRight_HandRightPlaneXYAxisY', 
                'WristRight_HandRightKinectZAxis'
            ]
            all_features.update({'body_' + k: v for k, v in zip(keys, body_posture_features)})
        
        # Forzar la conversión de cada valor a un tipo serializable simple (número o cadena)
        for k, v in all_features.items():
            if not isinstance(v, (int, float, str)):
                try:
                    all_features[k] = float(v)
                except Exception:
                    all_features[k] = str(v)
        
        return all_features