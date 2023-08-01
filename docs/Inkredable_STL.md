# Inkredable 
script utility to generate a parametric orthesis 3D object
to handle a pen using blender >= 2.82 in command line mode

## Pre-requisite :
- Blender >= 2.82
- Python >= 3.5

## Install :
unzip Inkredable.zip to the directory of your choice MY_DIR
You get the following hierarchy :
```
    MY_DIR/Inkredable/scripts/empty.blend
                            Inkredable.py
                            Geom.py
    MY_DIR/Inkredable/in/default.json -> example of measurements to provide
    MY_DIR/Inkredable/out/default.STL -> corresponding 3D model file
    MY_DIR/Inkredable/InkredableMeasurements.png
    MY_DIR/Inkredable/README.md
```
## Usage :
to generate your made to measure orthesis :
- Take your measurements as described on the image InkredableMeasurements.png,
- Edit the file default.json in the in/ directory, put your own values and save as name.json in the same directory
- In a shell terminal, execute the following command line

### On windows environment
`
blender.exe --background MY_DIR\Inkredable\scripts\empty.blend --python MY_DIR\Inkredable\scripts\Inkredable.py -- name
`
### On linux environment
`
blender --background MY_DIR/Inkredable/scripts/empty.blend --python MY_DIR/Inkredable/scripts/Inkredable.py -- name
`

- Note: if -- name not supplied, loads in/default.json if not supplied)
- name.STL 3D model is then generated in out/directory