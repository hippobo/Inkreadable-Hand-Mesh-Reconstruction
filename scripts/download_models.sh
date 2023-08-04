# Setup
export REPO_DIR=$PWD
echo "Working directory: ${REPO_DIR}"

if [ ! -d ${REPO_DIR}/models ] ; then
    mkdir -p ${REPO_DIR}/models
fi

BLOB='https://datarelease.blob.core.windows.net/metro'

# Download our pre-trained models
if [ ! -d ${REPO_DIR}/models/graphormer_release ] ; then
    mkdir -p ${REPO_DIR}/models/graphormer_release
fi

# Mesh Graphormer for hand mesh reconstruction (trained on FreiHAND)
wget -nc ${BLOB}/models/graphormer_hand_state_dict.bin -O ${REPO_DIR}/models/graphormer_release/graphormer_hand_state_dict.bin || true

# Download the ImageNet pre-trained HRNet models
# The weights are provided by https://github.com/HRNet/HRNet-Image-Classification
if [ ! -d ${REPO_DIR}/models/hrnet ] ; then
    mkdir -p ${REPO_DIR}/models/hrnet
fi

wget -nc ${BLOB}/models/hrnetv2_w64_imagenet_pretrained.pth -O ${REPO_DIR}/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth || true
wget -nc ${BLOB}/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O ${REPO_DIR}/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml || true
