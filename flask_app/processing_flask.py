import numpy as np
from trimesh.transformations import scale_matrix
import mediapipe as mp
import cv2
import torch
from flask_app.rendering_flask import  render_mesh_trimesh
import torch
from torchvision import transforms
from PIL import Image
import src.modeling.data.config as cfg



mp_hands = mp.solutions.hands


transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


def run_inference(image_list, Graphormer_model, mano, mesh_sampler):
    
    Graphormer_model.eval()
    mano.eval()
    
    with torch.no_grad():
        for image_file in image_list:
            
            if 'pred' not in image_file:
                att_all = []
                print(image_file)
                img = Image.open(image_file)
                img_tensor = transform(img)
                img_visual = transform_visualize(img)

                batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
                # forward-pass
                pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = Graphormer_model(batch_imgs, mano, mesh_sampler)
                # obtain 3d joints from full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
                pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
                pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
                
                # save attention
                att_max_value = att[-1]
                att_cpu = np.asarray(att_max_value.cpu().detach())
                att_all.append(att_cpu)

                # obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
                # obtain 2d joints, which are projected from 3d joints of mesh
                hand_verts = pred_vertices.cpu().detach()
                
                hand_joints = pred_3d_joints_from_mesh.cpu().detach()

                mano_layer = mano.get_layer()

                mano_faces = mano_layer.th_faces.numpy()

                hand_verts = hand_verts[0].numpy()
                hand_joints = hand_joints[0].numpy()

                np.save("./samples/hand_info_export/hand_verts.npy", hand_verts)
                np.save("./samples/hand_info_export/hand_joints.npy", hand_joints)
                np.save("./samples/hand_info_export/hand_faces.npy", mano_faces)


def normalize_vertices_and_joints(vertices, joints): 
    # Compute the global min and max 
    min_val = np.min((vertices))
    max_val = np.max((vertices))
   
    vertices = (vertices ) / (max_val - min_val)
    joints = (joints ) / (max_val - min_val)
   
    return vertices, joints

def uv_to_xy(uv, K ,z):
    uv_copy = uv.copy()
    uv_copy *= 224
    x = z * (uv_copy[0] - K[0,2]) /K[0,0]

    y = z * (uv_copy[1] - K[1,2]) / K[1,1]
    return np.array([x,y])
            
def detect_mediapipe_2d(img_fp, isRender=True):
    joint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1) as hands:
        
        # Read an image, flip it around y-axis for correct handedness output (see above).
        image = cv2.flip(cv2.imread(img_fp), 1)

        # Convert the BGR image to RGB before processing.
        if not isRender:
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            print("No hand landmarks detected")
            return None
        
        hand_landmarks = results.multi_hand_landmarks
        joint_coords = []

        if hand_landmarks:
            hand_landmark = hand_landmarks[0]
            for joint_num, lm in enumerate(hand_landmark.landmark):
                        if joint_num in joint_indices:
                            x,y = lm.x, lm.y
                            joint_coords.append((x, y))

        #np.save('./samples/hand_info_export/mediapipe_2d_mesh.npy', joint_coords)
        mediapipe_2d_mesh = np.array(joint_coords)
        mediapipe_2d_mesh[:, 0] = 1 - mediapipe_2d_mesh[:, 0]

        return mediapipe_2d_mesh

def get_joint_angles(mediapipe_3d_coords):
    joint_list = [[0, 5, 8], [0, 9, 12], [0, 13, 16], [0, 17, 20]]
    angles = np.array([])
    isClosed = False
    
    for idx, joint in enumerate(joint_list):
        a = np.array(mediapipe_3d_coords[joint[0]])
        b = np.array(mediapipe_3d_coords[joint[1]])
        c = np.array(mediapipe_3d_coords[joint[2]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        np.append(angles,angle)
        
        if np.mean(np.degrees(angle)) < 100:
            isClosed = True

        

        # print("joint angles ", idx, np.degrees(angle))

    return isClosed

def reprojection_error_minimize_scalar(scale,translation_z,wrist_translation, K, mediapipe_2d_image, joints):
    # scale is the scalar
    
    joints = scale * joints
    joints = joints + [wrist_translation[0],wrist_translation[1],translation_z]
    
    mano_2D_homogeneous = (K @ joints.T).T

    # Convert back from homogeneous to regular coordinates
    mano_2D = mano_2D_homogeneous[:, :2] / (mano_2D_homogeneous[:, 2:] + 0.00000001)

    mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
    mediapipe_2d_denorm_image[:, 0] = mediapipe_2d_image[:, 0] * 224 
    mediapipe_2d_denorm_image[:, 1] = mediapipe_2d_image[:, 1] * 224

    # print("scale", scale)
    # print("scale error: ", np.linalg.norm(mano_2D - mediapipe_2d_denorm_image))
    
    
    return np.sum((mano_2D - mediapipe_2d_denorm_image)**2)


def reprojection_error_wrist(translation_vector, translation_z, K,  mediapipe_2d_image, joints):

    
    translation_vector = np.append(translation_vector, translation_z)
    
    
    wrist = joints[0] + translation_vector
    mano_2D_homogeneous = (K @ wrist.T).T

    

    # Convert back from homogeneous to regular coordinates
    mano_wrist_2D = mano_2D_homogeneous[:2] / mano_2D_homogeneous[2]

    mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
    mediapipe_2D_wrist = mediapipe_2d_denorm_image[0] * 224

    return np.sum((mano_wrist_2D - mediapipe_2D_wrist)**2)



 
def minimize_2d_mediapipe_wrist(translation_vector, translation_z, mesh_scaled, K,mediapipe_2d_image, mediapipe_2d_mesh):


    translation_vector_xyz = np.append(translation_vector, translation_z)
    

   
    # translation =  translation_matrix(translation_vector_xy)

    mesh_translated = mesh_scaled.copy()
    print("translation_vector_xyz", translation_vector_xyz)
    # mesh_translated = mesh.apply_transform(translation)
    if mediapipe_2d_mesh is None:
        return 10000
    
    render_mesh_trimesh(mesh_translated,K,translation_vector_xyz[0], translation_vector_xyz[1], translation_vector_xyz[2])

    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')
    if mediapipe_2d_mesh is None:
        return 10000
   
        
    
    mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
    mediapipe_2d_denorm_image_wrist = mediapipe_2d_denorm_image[0] * 224 
    

    mediapipe_2d_denorm_mesh = mediapipe_2d_mesh.copy()
    mediapipe_2d_denorm_mesh_wrist = mediapipe_2d_denorm_mesh[0] * 224 

   
    print("wrist error", np.linalg.norm(mediapipe_2d_denorm_mesh_wrist - mediapipe_2d_denorm_image_wrist))

    return np.sum((mediapipe_2d_denorm_mesh_wrist - mediapipe_2d_denorm_image_wrist)**2)


def mediapipe_error_minimize_scalar(scale, mesh, wrist_pos_image, translation_vector_xyz, K, mediapipe_2d_mesh, mediapipe_2d_image):


    scaling = scale_matrix(scale)
    mesh_scaled = mesh.copy()
    mesh_scaled = mesh_scaled.apply_transform(scaling)
    translation_z = translation_vector_xyz[2] - np.max(mesh_scaled.vertices[:, 2])


    render_mesh_trimesh(mesh_scaled, K, translation_vector_xyz[0], translation_vector_xyz[1],translation_z) 
    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')

     
    if mediapipe_2d_mesh is None:
        return 10000
    wrist_pos_mesh = uv_to_xy(mediapipe_2d_mesh[0],K, z=translation_z) 
    wrist_translation = 2 * wrist_pos_image - wrist_pos_mesh

    render_mesh_trimesh(mesh_scaled,K,wrist_translation[0], wrist_translation[1], translation_z)

    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')

    
    if mediapipe_2d_mesh is None:
        return 10000
    else: 

        
       
        mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
        mediapipe_2d_denorm_image[:, 0] = mediapipe_2d_image[:, 0] * 224 
        mediapipe_2d_denorm_image[:, 1] = mediapipe_2d_image[:, 1] * 224

        mediapipe_2d_denorm_mesh = mediapipe_2d_mesh.copy()
        mediapipe_2d_denorm_mesh[:, 0] = mediapipe_2d_mesh[:, 0] * 224 
        mediapipe_2d_denorm_mesh[:, 1] = mediapipe_2d_mesh[:, 1] * 224 

        
        print("scale" , scale)
       
        print("scale error", np.linalg.norm(mediapipe_2d_denorm_mesh - mediapipe_2d_denorm_image))
        print("translation z", translation_z) 
        print("wrist translation", wrist_translation)
        

        return np.sum((mediapipe_2d_denorm_mesh - mediapipe_2d_denorm_image)**2)

def minimize_rotation(rotatex,rotatey, scale, mesh, translation_vector_xyz, K, mediapipe_2d_mesh, mediapipe_2d_image):
    scaling = scale_matrix(scale)

    mesh_scaled = mesh.copy()
    mesh_scaled = mesh_scaled.apply_transform(scaling)
    translation_z = translation_vector_xyz[2] - np.max(mesh_scaled.vertices[:, 2])
    render_mesh_trimesh(mesh_scaled,K,translation_vector_xyz[0], translation_vector_xyz[1], translation_z, rotatex_angle=rotatex, rotatey_angle=rotatey)
    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')
    if mediapipe_2d_mesh is None:
        return 10000
    else: 

        
       
        mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
        mediapipe_2d_denorm_image[:, 0] = mediapipe_2d_image[:, 0] * 224 
        mediapipe_2d_denorm_image[:, 1] = mediapipe_2d_image[:, 1] * 224

        mediapipe_2d_denorm_mesh = mediapipe_2d_mesh.copy()
        mediapipe_2d_denorm_mesh[:, 0] = mediapipe_2d_mesh[:, 0] * 224 
        mediapipe_2d_denorm_mesh[:, 1] = mediapipe_2d_mesh[:, 1] * 224 

        
        print("rotatex" , rotatex)
        print("rotatey" , rotatey)
       
        print("rotation error", np.linalg.norm(mediapipe_2d_denorm_mesh - mediapipe_2d_denorm_image))
        
        

        return np.sum((mediapipe_2d_denorm_mesh - mediapipe_2d_denorm_image)**2)