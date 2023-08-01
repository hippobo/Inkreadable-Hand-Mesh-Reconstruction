"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D hand mesh reconstruction from an image
"""


#methods written are in src.utils.processing, vizualiztion and rendering

from __future__ import absolute_import, division, print_function
import subprocess
import shlex
import argparse
import os
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
from src.utils.visualization import  run_hand_detection, run_hand_detection_fp
from src.utils.rendering import render_mesh_trimesh
   
from src.utils.processing import normalize_vertices_and_joints,uv_to_xy,detect_mediapipe_2d, mediapipe_error_minimize_scalar, get_joint_angles


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
from torch.nn.functional import normalize
from scipy.optimize import least_squares, minimize, minimize_scalar, basinhopping
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
import pyrender
from shapelysmooth import taubin_smooth


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

surface = None


def get_joints(label_name):
    
   
        
        if label_name == 'Thumb Base':
            chosen_joints = ['Thumb_2' ,'Thumb_3']
        if label_name == 'Thumb Extremity':
            chosen_joints = ['Thumb_3', 'Thumb_4']
        if label_name == 'Index Base':
            chosen_joints = ['Index_2', 'Index_3']
        if label_name == 'Index Extremity':
            chosen_joints = ['Index_3', 'Index_4']

        return chosen_joints
          


def update_plane(center,normal):
    global surface
    
    plane_size = (ax1.get_xlim()[1] - ax1.get_xlim()[0])/10
   
   

    if surface is not None:
        surface.remove()

    d = -center.dot(normal)
    xx, yy = np.meshgrid(np.linspace(center[0]-plane_size,center[0]+plane_size,10), 
                         np.linspace(center[1]-plane_size,center[1]+plane_size,10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
  

    surface = ax1.plot_surface(xx, yy, z, color='red', alpha=0.8)

    return surface
    
    

        

multi_slice = None  # Initialize a global variable for the slices

def calculate_slices(chosen_joints):
    global multi_slice  # We will modify the global variable

    normal = zip_joint_names[chosen_joints[1]] - zip_joint_names[chosen_joints[0]]
    fixed_center = zip_joint_names[chosen_joints[0]]
    heights = np.linspace(0.0, np.linalg.norm(normal),  100)

    multi_slice = mesh_vizualization.section_multiplane(plane_origin=fixed_center, 
                                                        plane_normal=normal, heights=heights)

def update_cross_section(val):
    global multi_slice  # We will read the global variable

    val_plane = val / 100.0
    chosen_joints = get_joints(radio_buttons.value_selected)

    ax2.clear()
    ax2.set_title('Finger Cross Section')

    normal = zip_joint_names[chosen_joints[1]] - zip_joint_names[chosen_joints[0]]
    center = zip_joint_names[chosen_joints[0]] + val_plane * normal
    
    update_plane(center, normal)

    if (multi_slice is not None):
        if val > 0:
            slice = multi_slice[val - 1]
        else: slice = multi_slice[0]
        poly_union = shapely.geometry.MultiPolygon(poly for poly in slice.polygons_full) 

        min_distance = float('inf')
        closest_polygon = None

        for i, geom in enumerate(poly_union.geoms):
            #Polygon smoothing
            geom = taubin_smooth(geometry=geom, factor=0.5, mu=-0.5, steps=25)

            xs, ys = geom.exterior.xy

            ax2.fill(xs, ys, alpha=0.5, ec='none')               
            # Calculate the perimeter in millimeters
            perimeter = geom.length

            # Get the centroid of the polygon
            centroid = geom.centroid
            
            # Calculate distance from origin
            distance = np.sqrt(centroid.x ** 2 + centroid.y ** 2)

            # If this polygon is closer to origin than previous closest, update closest_polygon
            if distance < min_distance:
                min_distance = distance
                closest_polygon = geom

            # Add text to the plot
            ax2.text(centroid.x, centroid.y, f'perim : {perimeter * 1000:.2f} mm', fontsize=8, ha='center')
        
        if closest_polygon is not None:
            centroid = closest_polygon.centroid
            ax2.text(centroid.x, centroid.y + 0.01, f' {radio_buttons.value_selected}', fontsize=10, color='red')
            
            

def on_radio_button_selected(label):
    chosen_joints = get_joints(label)
    calculate_slices(chosen_joints)
    update_cross_section(0)  # Initial update



def run_inference(image_list, Graphormer_model, mano, mesh_sampler):
# switch to evaluate mode
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
                pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())
                hand_verts = pred_vertices.cpu().detach()
              
                hand_joints = pred_3d_joints_from_mesh.cpu().detach()

                mano_layer = mano.get_layer()

                mano_faces = mano_layer.th_faces.numpy()

                hand_verts = hand_verts[0].numpy()
                hand_joints = hand_joints[0].numpy()

                np.save("./samples/hand_info_export/hand_verts.npy", hand_verts)
                np.save("./samples/hand_info_export/hand_joints.npy", hand_joints)
                np.save("./samples/hand_info_export/hand_faces.npy", mano_faces)
               
                #CAM PARAMS : scaling factor s + 2d translation vector (size 3)
                
                
                # visual_imgs_output = visualize_mesh(renderer, batch_visual_imgs[0],
                #                                             pred_vertices[0].detach(), 
                #                                         pred_camera.detach())
                

                # visual_imgs_output = visualize_mesh_and_attention( renderer, batch_visual_imgs[0],
                #                                             pred_vertices[0].detach(), 
                #                                             pred_vertices_sub[0].detach(), 
                #                                             pred_2d_coarse_vertices_from_mesh[0].detach(),
                #                                             pred_2d_joints_from_mesh[0].detach(),
                #                                             pred_camera.detach(),
                #                                             att[-1][0].detach())
               
                # visual_imgs = visual_imgs_output.transpose(1,2,0)
                # visual_imgs = np.asarray(visual_imgs)
                        
                # temp_fname = image_file[:-4] + '_graphormer_pred.jpg'
                # # print('save to ', temp_fname)
                # cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
    return



def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
   
    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = Mesh()
    

    # Load pretrained model
    trans_encoder =[ ]

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]
    
    # which encoder block to have graph convs
    which_blk_graph = [int(item) for item in args.which_gcn.split(',')]

    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)

    else:
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*2)

            if which_blk_graph[i]==1:
                config.graph_conv = True
                logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = args.mesh_type

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
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
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(config, backbone, trans_encoder)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(args.resume_checkpoint)
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
        setattr(_model.trans_encoder[-1].config,'device', args.device)

    _model.to(args.device)
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename) 
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(image_list, _model, mano_model, mesh_sampler)


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")       
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.")  
    parser.add_argument("--image_file_or_path", default='./samples/hand_cropped', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default="./models/graphormer_release/graphormer_hand_state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")  
    parser.add_argument("--which_gcn", default='0,0,1', type=str, 
                        help="which encoder block to have graph conv. Encoder1, Encoder2, Encoder3. Default: only Encoder3 has graph conv") 
    parser.add_argument("--mesh_type", default='hand', type=str, help="body or hand") 

    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=True, action='store_true',) 
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    
    args = parse_args()
    #run_hand_detection()
    run_hand_detection_fp("./samples/hand_uncropped/hand_image_uncropped.png")

    main(args)

    
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

    #mesh for visualization 
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
    print("scale factor", scale_factor)

    
    H = 224

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
    
    hand_image = cv2.imread("./samples/hand_cropped/hand_image_cropped.jpg")
    hand_mesh_image = cv2.imread('./samples/hand_rendered/rendered_image_trimesh.png')

    mediapipe_2d_mesh = np.array(mediapipe_2d_mesh)
    
    
    mediapipe_2d_denorm_image = mediapipe_2d_image.copy()
    mediapipe_2d_denorm_image[:, 0] = mediapipe_2d_image[:, 0] * 224 
    mediapipe_2d_denorm_image[:, 1] = mediapipe_2d_image[:, 1] * 224

    mediapipe_2d_denorm_mesh = mediapipe_2d_mesh.copy()
    mediapipe_2d_denorm_mesh[:, 0] = mediapipe_2d_denorm_mesh[:, 0] * 224 
    mediapipe_2d_denorm_mesh[:, 1] = mediapipe_2d_denorm_mesh[:, 1] * 224 
   
    mediapipe_2d_denorm_mesh_flipped = mediapipe_2d_denorm_mesh.copy()
    mediapipe_2d_denorm_mesh_flipped[:, 1] = H - mediapipe_2d_denorm_mesh_flipped[:, 1]

    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

    hand_mesh_image = cv2.cvtColor(hand_mesh_image, cv2.COLOR_BGR2RGB)
  

    print("final scale error of coords: ", np.linalg.norm(mediapipe_2d_denorm_mesh -mediapipe_2d_denorm_image), "with scale factor :", scale_factor)
    print("depth coin: ", depth_coin * 1000, " mm ")

    joint_names = ['Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4']

    zip_joint_names = dict(zip(joint_names, joints))

    fig = plt.figure(figsize=(18,12))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Hand Mesh')  
    plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(left=0.10)
    ax_radio= plt.axes([0.025, 0.2, 0.15, 0.15], facecolor='lightgoldenrodyellow')
    labels = ['Thumb Base', 'Thumb Extremity', 'Index Base', 'Index Extremity']
    radio_buttons = RadioButtons(ax_radio, labels)

    ax_slider = plt.axes([0.20, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider= Slider(ax_slider, 'Slide along joint', valmin=0, valmax=100, valinit=0,valstep=1)


    mesh_3D = Poly3DCollection(mesh_vizualization.vertices[mesh_vizualization.faces], alpha=0.2)


    face_color = (1.0, 1.0, 0.9)
    edge_color = (0, 0, 0)
    mesh_3D.set_edgecolor(edge_color)
    mesh_3D.set_facecolor(face_color)
    ax1.add_collection3d(mesh_3D)
    ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)

    all_points = np.concatenate((mesh_vizualization.vertices, joints), axis=0)

    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])

    

    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')

    fig2 = plt.figure(figsize=(18,12))

  
    mediapipe_2d_denorm_image_flipped = mediapipe_2d_denorm_image.copy()
    mediapipe_2d_denorm_image_flipped[:, 1] = H - mediapipe_2d_denorm_image[:, 1]

    
    ax3 = fig2.add_subplot(121)
    ax3.set_title('Projected points Mediapipe Image and Mesh')
    ax3.set_xlim([0, 224])
    ax3.set_ylim([0, 224])
    

    ax3.scatter(mediapipe_2d_denorm_image_flipped[:, 0], mediapipe_2d_denorm_image_flipped[:, 1], color='b', label='Mediapipe Base Image')
    for i in range(len(mediapipe_2d_denorm_image_flipped)):
        ax3.annotate(str(i), (mediapipe_2d_denorm_image_flipped[i, 0], mediapipe_2d_denorm_image_flipped[i, 1]))

    x_lim = ax3.get_xlim()
    y_lim = ax3.get_ylim()

    ax3.imshow(hand_image, aspect='auto', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], zorder=0)
    # Adjust the y-ticks and create new labels for them
    yticks = ax3.get_yticks()
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(H - yticks)


    ax3.scatter(mediapipe_2d_denorm_mesh_flipped[:, 0], mediapipe_2d_denorm_mesh_flipped[:, 1], color='r', label = 'Mediapipe mesh')
   
    for i in range(len(mediapipe_2d_denorm_mesh_flipped)):
        ax3.annotate(str(i), (mediapipe_2d_denorm_mesh_flipped[i, 0], mediapipe_2d_denorm_mesh_flipped[i, 1]))

    ax4 = fig2.add_subplot(122)
    ax4.set_title('Mediapipe mesh coordinates')
    ax4.set_xlim([0, 224])
    ax4.set_ylim([0, 224])
 
    ax4.scatter(mediapipe_2d_denorm_mesh_flipped[:, 0], mediapipe_2d_denorm_mesh_flipped[:, 1], color='r', label = 'Mediapipe mesh')
    
    # Get scatter plot limits
    x_lim = ax4.get_xlim()
    y_lim = ax4.get_ylim()

    # Display the image according to scatter plot limits
    ax4.imshow(hand_mesh_image, aspect='auto', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], zorder=0)
    
    for i in range(len(mediapipe_2d_denorm_mesh_flipped)):
        ax4.annotate(str(i), (mediapipe_2d_denorm_mesh_flipped[i, 0], mediapipe_2d_denorm_mesh_flipped[i, 1]))

    yticks = ax4.get_yticks()
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(H - yticks)

    ax3.legend()
    ax4.legend()
    ax3.set_aspect('equal', 'box')
    ax4.set_aspect('equal', 'box')
   
    slider.on_changed(update_cross_section)
    on_radio_button_selected(radio_buttons.value_selected)

    radio_buttons.on_clicked(on_radio_button_selected)
    


    plt.show()





