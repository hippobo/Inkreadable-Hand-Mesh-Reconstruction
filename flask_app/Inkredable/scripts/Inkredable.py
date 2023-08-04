# Inkredable.py
# script utility to generate a parametric orthesis 3D object to handle a pen using blender >= 2.82
# Author : L. Boissieux (c) Inria
# Date 09/05/2023
# Version 1.0

# Install : unzip Inkredable.zip to the directory of your choice MY_DIR
# You get the following hierarchy :
# MY_DIR/Inkredable/scripts/empty.blend
#                           Inkredable.py
#                           Geom.py
# MY_DIR/Inkredable/in/default.json -> example of measurements to provide
# MY_DIR/Inkredable/out/default.STL -> corresponding 3D model file
#
# Usage on windows environment
# blender.exe --background MY_DIR\Inkredable\scripts\empty.blend --python MY_DIR\Inkredable\scripts\Inkredable.py [-- name]
#
# Usage on linux environment
# blender --background MY_DIR/Inkredable/scripts/empty.blend --python MY_DIR/Inkredable/scripts/Inkredable.py [-- name]
#
# name(optional) : orthesis name, will load name.json from in directory and will generate name.STL in out directory
#                  (load  in/default.json if not supplied)
#                  don't forget -- before name !

import bpy
import bmesh
import math
import mathutils
import os
import sys
# Json config
import json

class Inkredable():
    def __init__(self):
        self.RHanded = True
        self.Txc = 60.0  #Thumb contour extremity
        self.Tbc = 62.0  #Thumb contour base
        self.Ixc = 50.0  #Index contour extremity
        self.Ibc = 52.0  #Index contour base
        self.H = 15.0    #height
        self.Pr = 4.0    #pen radius
        self.Tk = 2.0    #thickness
        self.TIa = math.radians(60.0)  #Thumb-Index angle
        self.IGa = math.radians(22.0)  #Index-Ground angle
        self.Ih = 3      #Interstice height
        #Useful intermediary variables
        self.HH = self.H/2.0
        self.Txr = self.Txc/(2*math.pi)  #Thumb extremity radius
        self.Tbr = self.Tbc/(2*math.pi)  #Thumb base radius
        self.Ixr = self.Ixc/(2*math.pi)  #Index extremity radius
        self.Ibr = self.Ibc/(2*math.pi)  #Index base radius

    def setIntermediary(self):
        self.HH = self.H/2.0
        self.Txr = self.Txc/(2*math.pi)  
        self.Tbr = self.Tbc/(2*math.pi)  
        self.Ixr = self.Ixc/(2*math.pi)  
        self.Ibr = self.Ibc/(2*math.pi)  

    def readParametersFromFile(self, fileName):
        with open(fileName, 'r') as f:
            fromJson = json.load(f)
        if fromJson[1]["Hand"] == "Left":
            self.RHanded = False
        self.Txc = float(fromJson[2]["Thumb contour extremity"])
        self.Tbc = float(fromJson[2]["Thumb contour base"])
        self.Ixc = float(fromJson[2]["Index contour extremity"])
        self.Ibc = float(fromJson[2]["Index contour base"])
        self.H = float(fromJson[2]["Height"])
        self.Pr = float(fromJson[2]["Pen radius"])
        self.Tk = float(fromJson[2]["Thickness"])
        self.TIa = math.radians(float(fromJson[2]["Thumb-Index angle"]))
        self.IGa = math.radians(float(fromJson[2]["Index-Ground angle"]))
        self.Ih = float(fromJson[2]["Interstice height"])
        self.setIntermediary()

def makeRing(Name, R1, R2, H, a):
    vs = [(0, 0, 0),(0,0,H),(0,-R1,0),(0,-R2,H)]
    fs = [(0,1,3,2)]
    nm = bpy.data.meshes.new('nm')
    nm.from_pydata(vs, [], fs)
    nm.update(calc_edges=True)
    obj = bpy.data.objects.new(Name, nm)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)  
    mesh = bpy.context.object.data
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.ops.mesh.spin(steps=30, angle=a, center=(0, 0, 0), axis=(0, 0, 1))
    bpy.ops.mesh.select_mode(type='VERT', action='TOGGLE')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.select_all(action='DESELECT')    
    bm = bmesh.from_edit_mesh(mesh)
    for v in bm.verts:
        if v.co.x == 0 and v.co.y == 0:
            v.select_set(True)
    bmesh.update_edit_mesh(mesh, True)      
    bpy.ops.mesh.delete(type='VERT')   
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.faces_shade_smooth()
    if a < 0:
        bpy.ops.mesh.flip_normals()
    bpy.ops.mesh.select_all(action='DESELECT')      
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)    
    return obj

def getRingExtremitiesPT(obj):
    p0 = obj.matrix_world @ obj.data.vertices[0].co
    p1 = obj.matrix_world @ obj.data.vertices[1].co
    p2 = obj.matrix_world @ obj.data.vertices[61].co
    p3 = obj.matrix_world @ obj.data.vertices[60].co
    p4 = obj.matrix_world @ obj.data.vertices[2].co
    p5 = obj.matrix_world @ obj.data.vertices[3].co
    p6 = obj.matrix_world @ obj.data.vertices[59].co
    p7 = obj.matrix_world @ obj.data.vertices[58].co
    pts=[p0, p1,p2,p3]
    tgs=[4*p0 - 3*p4, 4*p1 - 3*p5,4*p2 - 3*p6,4*p3 - 3*p7]
    return pts, tgs
    
def convertConnectShapeToMesh(obj):
    obj.select_set(True) 
    bpy.ops.object.convert(target='MESH')
    mesh = obj.data
    nbPS =  int(len(mesh.vertices) / 4)
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    for i in range (0, nbPS-1):
        bm.faces.new((bm.verts[i], bm.verts[nbPS + i], bm.verts[nbPS + i+1], bm.verts[i +1]))
        bm.faces.new((bm.verts[2*nbPS + i], bm.verts[3*nbPS + i], bm.verts[3*nbPS + i + 1], bm.verts[2*nbPS + i + 1]))   
    bmesh.update_edit_mesh(mesh, True)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.faces_shade_smooth()
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False) 


if __name__ == '__main__':

    # Get arguments
    argv = sys.argv
    # Setup base directory and files
    baseDir = os.path.dirname(argv[argv.index("--python") + 1])
    # Add Inkredable scripts path to Blender and load Geom module
    if not baseDir in sys.path:
        sys.path.append(baseDir)
    from Geom import *
    baseDir = os.path.dirname(baseDir) + '/'
    if len(argv) < 5:
        print ("Arguments missing: --background empty.blend --python script.py [-- name]")
        exit
    elif len(argv) == 5 : # no Json supplied
        name = "default"
    else :
        # get name after "--"
        name = argv[argv.index("--") + 1]
    JsonFilename = baseDir + "in/" + name + ".json"
    STLFilename = baseDir + "out/" + name + ".STL"

    # Get pen handler parameters
    IR = Inkredable()
    IR.readParametersFromFile(JsonFilename)

    # Set units to mm
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'  
    bpy.context.scene.unit_settings.scale_length = 0.001 

    # Make Thumb
    Thumb = makeRing("Thumb", IR.Txr, IR.Tbr, IR.H, 4.35)
    bpy.ops.transform.translate(value=(-IR.Txr-IR.Pr, -IR.Pr*(math.sin(IR.TIa))/2, IR.Txr))
    bpy.context.scene.tool_settings.transform_pivot_point = 'ACTIVE_ELEMENT'
    bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X')
    bpy.ops.object.select_all(action='DESELECT')
    # Make Index
    Index = makeRing("Index", IR.Ixr, IR.Ibr, IR.H, -4.71)
    bpy.context.scene.tool_settings.transform_pivot_point = 'ACTIVE_ELEMENT'
    bpy.ops.transform.translate(value=(IR.Ixr+IR.Pr,IR.Pr*(math.sin(IR.TIa))/2,IR.Ixr*math.cos(IR.IGa)))
    bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X')
    bpy.ops.transform.rotate(value=-IR.TIa, orient_axis='Y', orient_type='LOCAL')
    bpy.ops.transform.rotate(value=IR.IGa, orient_axis='X', orient_type='LOCAL')
    bpy.ops.transform.translate(value=(-((2*IR.Pr)+IR.Ixr)*(1-math.cos(IR.TIa))/2,((2*IR.Pr)+IR.Ixr)*math.sin(IR.TIa)/2,0))

    # Get rings extremities points and compute its tangents	
    ptsT, tgsT = getRingExtremitiesPT(Thumb)
    ptsI, tgsI = getRingExtremitiesPT(Index)  

    # Build spline shape to connect the rings :
    # made of 4 smooth splines of 3 points,
    # 1st and last being the rings extremities, as bezier corners using the computed tangents
    # middles ones being computed to match the prototype shape made by E. Moullet
    # create the Curve Datablock
    sps = bpy.data.curves.new('SplineShape', type='CURVE')
    sps.dimensions = '3D'
    sps.resolution_u = 6

    sp0 = sps.splines.new('BEZIER')
    sp0.resolution_u = 6
    sp0.bezier_points.add(2) # add 3 points but splines have already origin by default as first point
    sp0.bezier_points[0].handle_left = sp0.bezier_points[0].co = ptsT[0]
    sp0.bezier_points[0].handle_right = tgsT[0]
    sp0.bezier_points[1].co = (IR.Pr*math.cos(math.radians(45)), -IR.Pr*math.cos(math.radians(45)),0)
    sp0.bezier_points[1].handle_left_type = sp0.bezier_points[1].handle_right_type = 'AUTO'
    sp0.bezier_points[2].handle_left = tgsI[0]
    sp0.bezier_points[2].handle_right = sp0.bezier_points[2].co = ptsI[0]

    sp1 = sps.splines.new('BEZIER')
    sp1.resolution_u = 6
    sp1.bezier_points.add(2)
    sp1.bezier_points[0].handle_left = sp1.bezier_points[0].co = ptsT[1]
    sp1.bezier_points[0].handle_right = tgsT[1]
    C = mathutils.Vector(getXYCircleCenter(ptsI[1], ptsT[1], IR.TIa, True))
    R = mathutils.Vector(rotateXYPoint(C, ptsT[1], IR.TIa/2.0))
    sp1.bezier_points[1].co = R
    sp1.bezier_points[1].handle_left_type = sp1.bezier_points[1].handle_right_type = 'AUTO'
    sp1.bezier_points[2].handle_left = tgsI[1]
    sp1.bezier_points[2].handle_right = sp1.bezier_points[2].co = ptsI[1]

    sp2 = sps.splines.new('BEZIER')
    sp2.resolution_u = 6
    sp2.bezier_points.add(2)
    sp2.bezier_points[0].handle_left = sp2.bezier_points[0].co = ptsT[2]
    sp2.bezier_points[0].handle_right = tgsT[2]
    C = mathutils.Vector(getXYCircleCenter(ptsI[2], ptsT[2], IR.TIa, True))
    R = mathutils.Vector(rotateXYPoint(C, ptsT[2], IR.TIa/2.0))
    R.z = IR.Ih
    sp2.bezier_points[1].co = R
    sp2.bezier_points[1].handle_left_type = sp2.bezier_points[1].handle_right_type = 'AUTO'
    sp2.bezier_points[2].handle_left = tgsI[2]
    sp2.bezier_points[2].handle_right = sp2.bezier_points[2].co = ptsI[2]

    sp3 = sps.splines.new('BEZIER')
    sp3.resolution_u = 6
    sp3.bezier_points.add(2)
    sp3.bezier_points[0].handle_left = sp3.bezier_points[0].co = ptsT[3]
    sp3.bezier_points[0].handle_right = tgsT[3]
    sp3.bezier_points[1].co = (IR.Pr*math.cos(math.radians(45)), -IR.Pr*math.cos(math.radians(45)),IR.Ih)
    sp3.bezier_points[1].handle_left_type = sp3.bezier_points[1].handle_right_type = 'AUTO'
    sp3.bezier_points[2].handle_left = tgsI[3]
    sp3.bezier_points[2].handle_right = sp3.bezier_points[2].co = ptsI[3]
    # Enhance the point at the pinch to avoid later intersecting faces problem :
    # make handles horizontal and change length
    sp3.bezier_points[1].handle_left_type = sp3.bezier_points[1].handle_right_type = 'FREE'
    sp3.bezier_points[1].handle_right.z = sp3.bezier_points[1].co.z
    sp3.bezier_points[1].handle_left = 2*sp3.bezier_points[1].co - sp3.bezier_points[1].handle_right

    Shape = bpy.data.objects.new('Shape', sps)
    bpy.context.collection.objects.link(Shape)
    bpy.context.view_layer.objects.active = Shape
     
    # Convert Shape to Mesh and create missing faces
    convertConnectShapeToMesh(Shape)
     
    # Assemble the rings and the connecting part
    bpy.data.objects['Shape'].select_set(True)
    bpy.data.objects['Index'].select_set(True)
    bpy.data.objects['Thumb'].select_set(True)
    bpy.ops.object.join()

    # Weld the parts and smooth faces
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    # Add thickness by applying a Solidify modifier
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = IR.Tk
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Solidify")

    # Add chamfer to inner edges
    mesh = bpy.data.objects['Shape'].data
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.ops.mesh.select_mode(type='EDGE', action='TOGGLE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.edges.ensure_lookup_table()    
    bm.edges[0].select_set(True)    
    bm.edges[12].select_set(True)
    bmesh.update_edit_mesh(mesh, True)
    bpy.ops.mesh.loop_multi_select(ring=False)
    bpy.ops.mesh.bevel(offset_type='OFFSET', offset=2*IR.Tk/3, segments=4)

    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    # Apply hand side
    if not IR.RHanded:
        bpy.ops.transform.mirror(orient_type='GLOBAL', constraint_axis=(False, True, False))
    
    # Rename    
    bpy.data.objects['Shape'].name = name 
    # Save mesh to STL 
    bpy.ops.export_mesh.stl(filepath=STLFilename, use_selection=True)
    
