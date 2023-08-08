# Setup
export REPO_DIR=$PWD
echo "Working directory: ${REPO_DIR}"

#Create directories

if [ ! -d ${REPO_DIR}/samples/hand_cropped ] ; then
    mkdir -p ${REPO_DIR}/samples/hand_cropped
fi

if [ ! -d ${REPO_DIR}/samples/hand_info_export ] ; then
    mkdir -p ${REPO_DIR}/samples/hand_info_export
fi

if [ ! -d ${REPO_DIR}/samples/Chessboard_Images ] ; then
    mkdir -p ${REPO_DIR}/samples/Chessboard_Images
fi

if [ ! -d ${REPO_DIR}/samples/hand_rendered ] ; then
    mkdir -p ${REPO_DIR}/samples/hand_rendered
fi

if [ ! -d ${REPO_DIR}/samples/hand_uncropped ] ; then
    mkdir -p ${REPO_DIR}/samples/hand_uncropped
fi

if [ ! -d ${REPO_DIR}/models ] ; then
    mkdir -p ${REPO_DIR}/models
fi
