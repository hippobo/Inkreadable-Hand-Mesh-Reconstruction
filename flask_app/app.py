from __future__ import absolute_import, division, print_function

from flask import Flask, request, render_template, jsonify, Response, send_file

from src.utils.visualization import  run_hand_detection, run_hand_detection_fp
from src.utils.rendering import render_mesh_trimesh

from src.utils.processing import normalize_vertices_and_joints,uv_to_xy,detect_mediapipe_2d, mediapipe_error_minimize_scalar, get_joint_angles
import traceback
import subprocess
import shlex
import argparse
import os
import glob
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
from manopth import demo
from manopth.manolayer import ManoLayer
import gc
import numpy as np
import cv2
import mediapipe as mp
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection
from flask_app.visualization_flask import  run_hand_detection, run_hand_detection_fp, get_joints, run_chessboard_detection
from flask_app.rendering_flask import render_mesh_trimesh, render_flask, run_blender
from src.utils.camera_calibration import calibrate
from flask_app.processing_flask import normalize_vertices_and_joints,uv_to_xy,detect_mediapipe_2d, mediapipe_error_minimize_scalar, get_joint_angles, run_inference
from werkzeug.utils import secure_filename

import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
import mano
import trimesh
from trimesh.transformations import scale_matrix, translation_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, RadioButtons 
import shapely
from shapely.geometry import Point

from torch.nn.functional import normalize
from scipy.optimize import least_squares, minimize, minimize_scalar, basinhopping
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
import pyrender
from shapelysmooth import taubin_smooth
from threading import Thread
import plotly.graph_objs as go


if torch.cuda.is_available():
    print("Using GPU", torch.cuda.is_available())
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

app = Flask(__name__)

def get_default_args():
    return {
        "num_workers": 4,
        "img_scale_factor": 1,
        "image_file_or_path": './samples/hand_cropped',
        "model_name_or_path": 'src/modeling/bert/bert-base-uncased/',
        "resume_checkpoint": "./models/graphormer_release/graphormer_hand_state_dict.bin",
        "output_dir": 'output/',
        "config_name": "",
        "arch": 'hrnet-w64',
        "num_hidden_layers": 4,
        "hidden_size": -1,
        "num_attention_heads": 4,
        "intermediate_size": -1,
        "input_feat_dim": '2051,512,128',
        "hidden_feat_dim": '1024,256,64',
        "which_gcn": '0,0,1',
        "mesh_type": 'hand',
        "run_eval_only": True,
        "device": 'cuda',
        "seed": 88
    }


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
    path = 'Inkredable/out/default.STL'
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
        thread = Thread(target=run_hand_detection_fp(image_path))
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
    your_mesh = trimesh.load_mesh('./flask_app/Inkredable/out/default.STL')

    # Convert the mesh data to JSON
    data = {
        "vertices": your_mesh.vertices.tolist(),
        "faces": your_mesh.faces.tolist(),
        "normals": your_mesh.face_normals.tolist(),
        
    }

    return jsonify(data)  # Use jsonify to return a response with the application/json mimetype
@app.route("/get_json_data", methods=["GET", "POST"])
def get_json_data():
    json_path = './flask_app/Inkredable/in/default.json'
    
    if request.method == "POST":
        # Load the existing JSON data
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # The client will send data as a single object with key-values to update
        for key, value in request.json.items():
            # Find the entry that contains the key
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
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    thread = Thread(target=run_hand_detection)
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
        predict()
      

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
def predict():

    args_dict = get_default_args()
    global logger

    # Setup CUDA, GPU & distributed training
    args_dict["num_gpus"] = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args_dict["num_workers"])
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
   
    mkdir(args_dict["output_dir"])
    logger = setup_logger("Graphormer", args_dict["output_dir"], get_rank())
    set_seed(args_dict["seed"], args_dict["num_gpus"])
    logger.info("Using {} GPUs".format(args_dict["num_gpus"]))

    mano_model = MANO().to(args_dict["device"])
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = Mesh()

    trans_encoder = []

    input_feat_dim = [int(item) for item in args_dict["input_feat_dim"].split(',')]
    hidden_feat_dim = [int(item) for item in args_dict["hidden_feat_dim"].split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]

    which_blk_graph = [int(item) for item in args_dict["which_gcn"].split(',')]


    if args_dict["run_eval_only"]==True and args_dict["resume_checkpoint"]!=None and args_dict["resume_checkpoint"]!='None' and 'state_dict' not in args_dict["resume_checkpoint"]:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args_dict["resume_checkpoint"]))
        _model = torch.load(args_dict["resume_checkpoint"])

    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args_dict["config_name"] if args_dict["config_name"] \
                    else args_dict["model_name_or_path"])

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args_dict["hidden_size"] = hidden_feat_dim[i]
            args_dict["intermediate_size"] = int(args_dict["hidden_size"]*2)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args_dict["mesh_type"]

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = args_dict[param]
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)
        
        # create backbone model
        if args_dict["arch"]=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args_dict["arch"]=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args_dict["arch"]))
            backbone = models.__dict__[args_dict["arch"]](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(config, backbone, trans_encoder)

        if args_dict["resume_checkpoint"]!=None and args_dict["resume_checkpoint"]!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args_dict["resume_checkpoint"]))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(args_dict["resume_checkpoint"])
            _model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    # update configs to enable attention outputs
    setattr(_model.trans_encoder[-1].config,'output_attentions', True)
    setattr(_model.trans_encoder[-1].config,'output_hidden_states', True)
    _model.trans_encoder[-1].bert.encoder.output_attentions = True
    _model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_model.trans_encoder[-1].config,'device', args_dict["device"])

    _model.to(args_dict["device"])
    logger.info("Run inference")

    image_list = []
    if not args_dict["image_file_or_path"]:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args_dict["image_file_or_path"]):
        image_list = [args_dict["image_file_or_path"]]
    elif op.isdir(args_dict["image_file_or_path"]):
        # should be a path with images only
        for filename in os.listdir(args_dict["image_file_or_path"]):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args_dict["image_file_or_path"]+'/'+filename) 

    else:
        raise ValueError("Cannot find images at {}".format(args_dict["image_file_or_path"]))

    if (len(image_list) ==0):
        raise ValueError("No Hand Image found - Please Retake Picture")
    run_inference(image_list, _model, mano_model, mesh_sampler)

    render_flask()

    get_all_measurements()
    return jsonify({"message": "Done"})

@app.route('/count_images', methods=['GET'])
def count_images():
    import os
    img_dir = './samples/Chessboard_Images'
    num_files = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
    return {'count': num_files}




@app.route('/video_feed_hand_detection')
def video_feed_hand_detection():
    return Response(run_hand_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_calibration')
def video_feed_calibration():

    files = glob.glob('./samples/Chessboard_Images/*')
    for f in files:
        os.remove(f)
    return Response(run_chessboard_detection(),
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


def get_all_measurements():

    chosen_joints = ['Thumb_2' ,'Thumb_3','Thumb_3', 'Thumb_4','Index_2', 'Index_3','Index_3', 'Index_4']
    joint_names = ['Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4']

    vertices = np.load("./samples/hand_info_export/hand_verts.npy")
    faces = np.load("./samples/hand_info_export/hand_faces.npy")
    joints = np.load("./samples/hand_info_export/hand_joints.npy")

    zip_joint_names = dict(zip(joint_names, joints))


    #create Trimesh Object
    mesh_vizualization = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    normals = [np.array(zip_joint_names[chosen_joints[i+1]]) - np.array(zip_joint_names[chosen_joints[i]]) for i in range(0, len(chosen_joints), 2)]


    fixed_center = [zip_joint_names[chosen_joints[i]] for i in range(0, len(chosen_joints), 2)]
    heights = [np.linspace(0.0, np.linalg.norm(normal), 100) for normal in normals]
    
    
    multi_slices = [mesh_vizualization.section_multiplane(plane_origin=i, 
                                                         plane_normal=j, heights=k) for i, j, k in zip(fixed_center, normals, heights)]
    
    slice_polygons = []
    for multislice in multi_slices:
        for slice in multislice:
          
            slice_polygons.append(shapely.geometry.MultiPolygon([poly for poly in slice.polygons_full]))

    

    origin = Point(0, 0)
    closest_polygon_lengths = []

    for multi_polygon in slice_polygons:
        closest_polygon = None
        min_distance = float('inf')
        
        for polygon in multi_polygon.geoms:
            center = polygon.centroid
            distance = center.distance(origin)
            
            if distance < min_distance:
                min_distance = distance
                closest_polygon = polygon
                
        closest_polygon_lengths.append(closest_polygon.length * 1000)

    measurements = {'Thumb contour base' : np.mean(closest_polygon_lengths[95:105]), 
                    'Thumb contour extremity' : np.mean(closest_polygon_lengths[150:160]), 
                    'Index contour base' : np.mean(closest_polygon_lengths[295:305]), 
                    'Index contour extremity' : np.mean(closest_polygon_lengths[350:360]) 
                   }

    # Parse the JSON file
    with open('./flask_app/Inkredable/in/default.json', 'r') as f:
        data = json.load(f)

    additional_measurements = data[2]  # assuming the measurements are in the 3rd item of the list in your JSON

    # Replace the values in the JSON with the computed measurements
    for key in measurements.keys():
        if key in additional_measurements:
            additional_measurements[key] = measurements[key]

    # Write the changes back to the JSON file
    with open('./flask_app/Inkredable/in/default.json', 'w') as f:
        json.dump(data, f, indent=3)

   
if __name__ == "__main__":
  app.run(debug=False, host='0.0.0.0', port=5000)
    
    





