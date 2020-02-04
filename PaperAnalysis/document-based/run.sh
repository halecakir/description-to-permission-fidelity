export ENCODER_DIR=$1
export ENCODER_TYPE=$2

parallel --jobs 10 < permissions.sh

