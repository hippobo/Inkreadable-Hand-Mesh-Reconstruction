#!/bin/bash

export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/Blender ] ; then
    mkdir -p $REPO_DIR/Blender
fi

# Change to the created directory
cd $REPO_DIR/Blender

# Define URL for the Blender 2.82
url="https://download.blender.org/release/Blender2.82/blender-2.82-linux64.tar.xz"

# Define the name of the file to be downloaded
file="blender-2.82-linux64.tar.xz"

# Download the file
wget $url -O $file

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful"

    # Extract the .tar.xz file
    tar -xf $file

    # Check if the extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction successful"

        # Delete the .tar.xz file
        rm $file

        # Check if the deletion was successful
        if [ $? -eq 0 ]; then
            echo "Deletion successful"
        else
            echo "Deletion failed"
        fi
    else
        echo "Extraction failed"
    fi
else
    echo "Download failed"
fi

# Change back to the original directory
cd $REPO_DIR
