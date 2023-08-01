import pyrender
import numpy as np
import subprocess   
from matplotlib import pyplot as plt


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


def run_blender():
    
    blend_file_path = " src/utils/Inkredable/scripts/empty.blend"
    blender_file_path = "Blender/blender-2.82-linux64/blender"
    script_file_path = "src/utils/Inkredable/scripts/Inkredable.py"

    command = [blender_file_path, "--background", blend_file_path, "--python", script_file_path]
    subprocess.run(command)