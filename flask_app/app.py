from __future__ import absolute_import, division, print_function

from flask import Flask, request, render_template, jsonify, Response, send_file

import traceback
import os
import glob
import os.path as op
import json
import torch
from torchvision.utils import make_grid
import numpy as np

from src.utils.visualization import  run_hand_detection_flask, run_hand_detection_fp_flask, get_joints, run_chessboard_detection_flask, get_all_measurements_flask
from src.utils.rendering import render_flask
from src.utils.camera_calibration import calibrate
from src.tools.run_gphmer_handmesh_inference import predict
from src.utils.rendering import run_blender
from werkzeug.utils import secure_filename

import trimesh
import numpy as np
import shapely

from threading import Thread


if torch.cuda.is_available():
    print("Using GPU", torch.cuda.is_available())
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

app = Flask(__name__)



def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_3d_trace(mesh):
    # Create a 3D scatter plot trace for the mesh
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    i, j, k = np.array(mesh.faces).T
    return {
        "type": 'mesh3d',
        "x": x.tolist(),
        "y": y.tolist(),
        "z": z.tolist(),
        "i": i.tolist(),
        "j": j.tolist(),
        "k": k.tolist(),
        "color": 'lightpink',
        "opacity": 0.50,
    }

def create_joint_trace(joints):
    # Create a 3D scatter plot trace for the joints
    x, y, z = zip(*joints)
    return {
        "type": 'scatter3d',
        "mode": 'markers',
        "x": x,
        "y": y,
        "z": z,
        "marker": {
            "size": 4,    
            "color": 'rgb(0, 0, 255)',
            "opacity": 1.0  # Make the joints fully opaque
        },
    }

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/calibration')
def calibration_page():
    return render_template('calibration.html')

@app.route('/mesh_rendering')
def current_page():
    return render_template('mesh_rendering.html')



@app.route('/download_stl')
def download_stl():
    username = request.args.get('username', default='default', type=str)
    if username == "":
        username = "writing"
    path = './src/utils/Inkredable/out/default.STL'
    return send_file(path, as_attachment=True, download_name=f'{username}_orthosis.STL')

@app.route("/upload_image", methods=["POST"])
def upload_image():
    directories = ['./samples/hand_uncropped/*', './samples/hand_cropped/*', './samples/hand_info_export/*', './samples/hand_rendered/*']


    for directory in directories:
        files = glob.glob(directory)
        for f in files:
            os.remove(f)

    image = request.files["image"]
    
    
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join("./samples/hand_uncropped/", filename)
        
        image.save(image_path)
        
    
                
    else:
        return jsonify({"message": "Invalid file format - only images are allowed"}), 400
        
        
    

    try:
        thread = Thread(target=run_hand_detection_fp_flask(image_path))
        thread.start()

        thread.join()
    except Exception as e:
        print(e)
        return jsonify({"message": "Error processing image - No hand/Coin found"}), 500
    return jsonify({"message": "Image uploaded and processed successfully"})

@app.route("/render_orthosis")
def render_orthosis():
    thread = Thread(target=run_blender)
    thread.start()

    thread.join()

    # Load the STL file
    your_mesh = trimesh.load_mesh('./src/utils/Inkredable/out/default.STL')

    # Convert the mesh data to JSON
    data = {
        "vertices": your_mesh.vertices.tolist(),
        "faces": your_mesh.faces.tolist(),
        "normals": your_mesh.face_normals.tolist(),
        
    }

    return jsonify(data)  # Use jsonify to return a response with the application/json mimetype
@app.route("/get_json_data", methods=["GET", "POST"])
def get_json_data():
    json_path = './src/utils/Inkredable/in/default.json'
    
    if request.method == "POST":
        # Load the existing JSON data
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # The client will send data as a single object with key-values to update
        for key, value in request.json.items():
            # Skip "Lang" and "Hand" 
            if key in ["Lang", "Hand"]:
                continue
            
            # Find the entry that contains the key and update it
            for entry in data:
                if key in entry:
                    entry[key] = value
                    break

        # Save the modified JSON data
        with open(json_path, 'w') as file:
            json.dump(data, file)

    # Load and return the current JSON data for frontend display
    with open(json_path, 'r') as file:
        data = json.load(file)

    return jsonify(data)


@app.route("/upload_images_calibration", methods=["POST"])
def upload_images_calib():
    files = glob.glob('./samples/Chessboard_Images/*')
    for f in files:
        os.remove(f)
    try:
        files = request.files.getlist("chessboard_images")
        for image in files:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image_path = os.path.join("./samples/Chessboard_Images", filename)
                image.save(image_path)
            else:
                return jsonify({"message": "Invalid file format - only images are allowed"}), 400
        
        ret, mtx, dist, rvecs, tvecs = calibrate('./samples/Chessboard_Images/',0.024, 6,9)
        np.save("./samples/camera_params/camera_matrix", mtx)
        np.save("./samples/camera_params/distortion_coefficients", dist)
        np.save("./samples/camera_params/rvecs", rvecs)
        np.save("./samples/camera_params/tvecs", tvecs)
        np.save("./samples/camera_params/dist", dist)
    
        return jsonify({"message":  "Images uploaded successfully - Calibration Done"})
    
    except Exception as e:
        files = glob.glob('./samples/Chessboard_Images/*')  # get all files
        for f in files:
            os.remove(f)  # remove each file
        traceback.print_exc()  # This will print the traceback to the console, which is useful for debugging
        return jsonify({"message": f"{str(e)} - Please check chessboard format/visibility"}), 500

@app.route("/calibrate_feed", methods=["POST"])
def calibrate_route():

    ret, mtx, dist, rvecs, tvecs = calibrate('./samples/Chessboard_Images/',0.024, 6,9)
    np.save("./samples/camera_params/camera_matrix", mtx)
    np.save("./samples/camera_params/distortion_coefficients", dist)
    np.save("./samples/camera_params/rvecs", rvecs)
    np.save("./samples/camera_params/tvecs", tvecs)
    np.save("./samples/camera_params/dist", dist)

    files = glob.glob('./samples/Chessboard_Images/*')
    for f in files:
        os.remove(f)
    return jsonify({"message": "Calibration Done"})
      
picture_status = "" 

@app.route("/start_hand_detection")
def start_hand_detection():
    global picture_status
    thread = Thread(target=run_hand_detection_flask)
    thread.start()
    thread.join()
   
    return jsonify({"message": "Done"})

@app.route("/get_picture_status")
def get_picture_status():
    global picture_status
    print("pic stat",picture_status)
    return jsonify({"status": picture_status})

@app.route("/start_prediction")
def start_prediction():
    app.logger.info("Entered the function")
  

    try:
        prediction()
      

        return jsonify({"message": "Done"})
    
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
  

@app.route('/data')
def data():
    joint_names = ['Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4']

    vertices = np.load("./samples/hand_info_export/hand_verts.npy")
    faces = np.load("./samples/hand_info_export/hand_faces.npy")
    joints = np.load("./samples/hand_info_export/hand_joints.npy")

    zip_joint_names = dict(zip(joint_names, joints))

    i, j, k = np.array(faces).T

    return jsonify({
        "mesh": {
            "vertices": vertices.tolist(),
            "joints": joints.tolist(),
            "i": i.tolist(),
            "j": j.tolist(),
            "k": k.tolist()
        },
       
        "joints": joints.tolist(),
        "zip_joint_names": {str(k): v.tolist() for k, v in zip_joint_names.items()}
    })


@app.route('/orthosis_rendering')
def orthosis_render():
    return render_template('orthosis_rendering.html')



@app.route('/predict', methods=['POST'])
def prediction():

    predict()

    render_flask()

    get_all_measurements_flask()

    return jsonify({"message": "Done"})

@app.route('/count_images', methods=['GET'])
def count_images():
    import os
    img_dir = './samples/Chessboard_Images'
    num_files = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
    return {'count': num_files}




@app.route('/video_feed_hand_detection')
def video_feed_hand_detection():
    return Response(run_hand_detection_flask(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_calibration')
def video_feed_calibration():

    files = glob.glob('./samples/Chessboard_Images/*')
    for f in files:
        os.remove(f)
    return Response(run_chessboard_detection_flask(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/calculate_slices', methods=['POST'])
def calculate_slices():
    data = request.get_json()
    
    label_name = data['label_name']
   

    joint_names = ['Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4']


    vertices = np.load("./samples/hand_info_export/hand_verts.npy")
    faces = np.load("./samples/hand_info_export/hand_faces.npy")
    joints = np.load("./samples/hand_info_export/hand_joints.npy")

    zip_joint_names = dict(zip(joint_names, joints))


    #create Trimesh Object
    mesh_vizualization = trimesh.Trimesh(vertices=vertices, faces=faces)
    chosen_joints = get_joints(label_name)

    normal = np.array(zip_joint_names[chosen_joints[1]]) - np.array(zip_joint_names[chosen_joints[0]])
    fixed_center = zip_joint_names[chosen_joints[0]]
    heights = np.linspace(0.0, np.linalg.norm(normal), 100)
    heights_plane = np.linspace(0.0, 1, 100)
    

    center_array = np.array([fixed_center + i * normal for i in heights_plane]).tolist()

    multi_slices = mesh_vizualization.section_multiplane(plane_origin=fixed_center, 
                                                         plane_normal=normal, heights=heights)
    multi_slices_json = [[[[x, y] for x, y in zip(*geom.exterior.xy)] for geom in shapely.geometry.MultiPolygon(poly for poly in slice.polygons_full).geoms] for slice in multi_slices]

    normal_json = normal.tolist()
    
    
    return jsonify(normalVector=normal_json, multi_slices=multi_slices_json, planeCenters=center_array)



   
if __name__ == "__main__":
  app.run(debug=False, host='0.0.0.0', port=5000)
    
    





