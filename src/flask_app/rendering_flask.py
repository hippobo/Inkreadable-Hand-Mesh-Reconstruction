import pyrender
import numpy as np
import trimesh

from matplotlib import pyplot as plt

import subprocess

from trimesh.transformations import scale_matrix, translation_matrix
from scipy.optimize import basinhopping



def render_mesh_trimesh(mesh, K, translation_x, translation_y, translation_z, display=False):

    mesh_render = pyrender.Mesh.from_trimesh(mesh)
    mesh_translate = np.array([
    [1.0, 0.0, 0.0, translation_x ],
    [0.0, 1.0, 0.0, translation_y], 
    [0.0, 0.0, 1.0, translation_z ],  
    [0.0, 0.0, 0.0, 1.0]
])
   
    scene = pyrender.Scene(bg_color=(255,235,205))
    # rotatey = trimesh.transformations.rotation_matrix(
    #                         angle=np.radians(rotatey_angle),
    #                         direction=[0, 1, 0],    
                            
    #                         point=mesh_render.centroid)
    
    # rotatex = trimesh.transformations.rotation_matrix(
    #                         angle=np.radians(rotatex_angle),
    #                         direction=[1, 0, 0],    
                            
    #                         point=mesh_render.centroid)


    # combined_transform = np.dot(mesh_translate, np.dot(rotatex, rotatey))

    mesh_node = pyrender.Node(mesh=mesh_render, matrix=mesh_translate)
   
    


    scene.add_node(mesh_node)
    
    pl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

#     cam_pose = np.array([
#     [1.0, 0.0, 0.0, translation_x ],
#     [0.0, -1.0, 0.0, translation_y],
#     [0.0, 0.0, -1.0, -translation_z ],  
#     [0.0, 0.0, 0.0, 1.0]
# ])
    
    cam_pose = np.array([
    [1.0, 0.0, 0.0, 0.0 ],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0 ],  
    [0.0, 0.0, 0.0, 1.0]
])

    camera_node = pyrender.Node(camera=cam, matrix=cam_pose)
    scene.add_node(camera_node)
    
    scene.add(pl, pose=cam_pose)
    light1 = pyrender.PointLight(color=np.ones(3), intensity=0.9)
    light2 = pyrender.PointLight(color=np.ones(3), intensity=0.9)
    light3 = pyrender.PointLight(color=np.array([0.6, 0.6, 0.6]), intensity=0.9)

  
    scene.add(light1, pose=np.array([[-1, 0, 0, -300], [0, 1, 0, -200], [0, 0, -1, -200], [0, 0, 0, 1]]))
    scene.add(light2, pose=np.array([[-1, 0, 0, 800], [0, 1, 0, 10], [0, 0, -1, 300], [0, 0, 0, 1]]))
    scene.add(light3, pose=np.array([[-1, 0, 0, -500], [0, 1, 0, 500], [0, 0, -1, 1000], [0, 0, 0, 1]]))
    if display:
        pyrender.Viewer(scene, viewport_size=(1280,720),use_raymond_lighting=True)

    # Create the renderer
    r = pyrender.OffscreenRenderer(224,224)

    # Render the scene
    color, depth = r.render(scene)


#     # Save the image
    plt.imsave('./samples/hand_rendered/rendered_image_trimesh.png', color)

def render_flask():
    from src.flask_app.processing_flask import normalize_vertices_and_joints,uv_to_xy,detect_mediapipe_2d, mediapipe_error_minimize_scalar, get_joint_angles
    K_params = np.load('./samples/hand_info_export/new_K_matrix.npy')
    coin_coords = np.load('./samples/hand_info_export/coin_coords.npy')
    mediapipe_2d_image = np.load('./samples/hand_info_export/mediapipe_2d_image.npy')
    mediapipe_3d_world_coords = np.load('./samples/hand_info_export/3d_world_coords.npy')
   
    handIsClosed = get_joint_angles(mediapipe_3d_world_coords)

    print("hand is closed", handIsClosed)


    mediapipe_2d_image[:, 0] = 1 - mediapipe_2d_image[:, 0] #change origin from top right (mp) to top left

   
    new_aspect_ratio = K_params[0]
    x_min_crop, y_min_crop = K_params[1], K_params[2]

   
    radius_coin = coin_coords[2]
    
    
    coin_diameter = 25.75 #mm, 2 euros

    #Load camera matrix 
   
    
    base_K = np.load("./samples/camera_params/camera_matrix.npy")
    base_K_logitech = np.load("./samples/camera_params/camera_matrix_logitech.npy")

    focal_length_uncropped = ((base_K[0][0]+ base_K[1][1]) / 2) #mm to m
    focal_length_uncropped_logitech = ((base_K_logitech[0][0]+ base_K_logitech[1][1]) / 2) #mm to m

    K = base_K.copy()
    K_logitech = base_K_logitech.copy()


    K_logitech[0][2] -= x_min_crop
    K_logitech[1][2] -= y_min_crop

    K_logitech = K_logitech * new_aspect_ratio


    K_logitech[2,2] = 1
    
    K = K_logitech



    # K[0][2] -= x_min_crop
    # K[1][2] -= y_min_crop

    # K = K * new_aspect_ratio


    # K[2,2] = 1

   
    print(K)
    

    #coin depth is calculated using the uncropped picture, so the uncropped focal length is used for better accuracy
    depth_coin = ((coin_diameter) * (focal_length_uncropped_logitech   / (radius_coin * 2))) / 1000 #mm to m 

    vertices = np.load("./samples/hand_info_export/hand_verts.npy")
    faces = np.load("./samples/hand_info_export/hand_faces.npy")
    joints = np.load("./samples/hand_info_export/hand_joints.npy")


    vertices,joints = normalize_vertices_and_joints(vertices,joints) 

    #create Trimesh Object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_vizualization = mesh.copy()

    #mesh smoothing
    # pinned = mesh.bounds.copy()
    # distance, index = cKDTree(mesh.vertices).query(pinned)
    # lap=trimesh.smoothing.laplacian_calculation(mesh, equal_weight=False, pinned_vertices=index)
    # trimesh.smoothing.filter_mut_dif_laplacian(mesh,lamb=0.2,iterations=20,volume_constraint=True,laplacian_operator=None)
   

    #PREPARE MESH FOR RENDERING
    if handIsClosed:
        initial_guess = 0.1
    else :
        initial_guess = 0.2
    scaling_wrist =  scale_matrix(initial_guess)



    mesh_scaled = mesh.copy()
    #mesh_scaled.vertices = mesh_scaled.vertices * starting_wrist_scale
    mesh_scaled = mesh_scaled.apply_transform(scaling_wrist)
    translation_z = depth_coin -  np.max(mesh_scaled.vertices[:, 2])
    
    
    wrist_pos_image = uv_to_xy(mediapipe_2d_image[0],K, z=translation_z)


    translation_init = np.array([wrist_pos_image[0], wrist_pos_image[1]])
    
   

    #RENDERING

    render_mesh_trimesh(mesh_scaled, K, translation_init[0], translation_init[1],translation_z) 

  

    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')

    wrist_pos_mesh = uv_to_xy(mediapipe_2d_mesh[0],K, z=translation_z) 
    wrist_translation = 2 * wrist_pos_image - wrist_pos_mesh
    

    render_mesh_trimesh(mesh_scaled, K, wrist_translation[0], wrist_translation[1],translation_z)
    
    
    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')

    translation_vector_xyz = np.array([wrist_translation[0],wrist_translation[1],depth_coin])
    
    mesh_copy = mesh.copy()
    func = lambda scale: mediapipe_error_minimize_scalar(scale, mesh_copy,wrist_pos_image, translation_vector_xyz, K, mediapipe_2d_mesh, mediapipe_2d_image)

    # Setup the bounds for the minimizer in the form of a list of tuples
    minimizer_kwargs = {"method": "SLSQP", "bounds": [(0.10, 0.4)]}

    x0 = initial_guess
    
    niter = 50
    stepsize = 0.02
    scale_method = "basinhopping"
    if scale_method == "basinhopping":
        basin_result = basinhopping(func, x0=x0, minimizer_kwargs=minimizer_kwargs,       
                                    niter=niter, stepsize=stepsize)

        # Get the optimized scale factor
        scale_factor= basin_result.x[0]
    else:
        errors = []
        scales = np.linspace(0.15, 0.35, 50)
        for i in scales:
            errors.append(func(i))
        scale_factor = scales[np.argmin(errors)]
   
    scaling = scale_matrix(scale_factor)
    mesh = mesh.apply_transform(scaling)

    translation_z = depth_coin -  np.max(mesh.vertices[:, 2])


    render_mesh_trimesh(mesh, K, wrist_translation[0], wrist_translation[1],translation_z)
    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')
    

    wrist_pos_mesh = uv_to_xy(mediapipe_2d_mesh[0],K, z=translation_z) 
    wrist_translation = 2 * wrist_pos_image - wrist_pos_mesh

    render_mesh_trimesh(mesh,K,wrist_translation[0], wrist_translation[1], translation_z)

    mediapipe_2d_mesh =detect_mediapipe_2d('./samples/hand_rendered/rendered_image_trimesh.png')

   

    #VISUALIZATION
    joints = joints * scale_factor
    
    joints = joints + [wrist_translation[0],wrist_translation[1],translation_z]

    scaling = scale_matrix(scale_factor)
    translation_mesh_vizualization =  translation_matrix([wrist_translation[0],wrist_translation[1],translation_z])

    mesh_vizualization = mesh_vizualization.apply_transform(scaling)

    mesh_vizualization = mesh_vizualization.apply_transform(translation_mesh_vizualization)

    np.save("./samples/hand_info_export/hand_verts.npy", mesh_vizualization.vertices)
    np.save("./samples/hand_info_export/hand_faces.npy", mesh_vizualization.faces)
    np.save("./samples/hand_info_export/hand_joints.npy", joints)
    

 
def run_blender():
    
    blend_file_path = " src/utils/Inkredable/scripts/empty.blend"
    blender_file_path = "Blender/blender-2.82-linux64/blender"
    script_file_path = "src/utils/Inkredable/scripts/Inkredable.py"

    command = [blender_file_path, "--background", blend_file_path, "--python", script_file_path]
    subprocess.run(command)