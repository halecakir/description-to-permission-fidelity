#!/bin/bash

PERMISSION_TYPE=$1
ENCODER_DIR=$2
ENCODER_TYPE=$3
REVIEW_OPTION=$4

EXPERIMENT_TYPE=$ENCODER_DIR-$ENCODER_TYPE

MODEL_TYPE="DocumentOnlyAttentionDyNET-REVIEW"
OUTPUT_DIR="../output/$MODEL_TYPE/$EXPERIMENT_TYPE-$REVIEW_OPTION"
PARAMETERS_DIR="$SECURITY_DATASETS/saved-parameters"

# Create output directory if not exists
mkdir -p $OUTPUT_DIR
OUT_FILE=$OUTPUT_DIR/$PERMISSION_TYPE.out
rm -f $OUT_FILE

python runner.py 	--permission-type $PERMISSION_TYPE \
					--saved-data $PARAMETERS_DIR/saved-data/ac-net/embeddings-documents-w2i-reviews.pickle \
					--saved-reviews $PARAMETERS_DIR/saved-data/reviews.pickle \
					--saved-predicted-reviews $PARAMETERS_DIR/saved-data/predicted-$PERMISSION_TYPE-reviews.pickle \
					--model-checkpoint $PARAMETERS_DIR/saved-models/$MODEL_TYPE-$EXPERIMENT_TYPE-$PERMISSION_TYPE.model \
					--outdir $OUT_FILE \
					--encoder-dir $ENCODER_DIR \
					--encoder-type $ENCODER_TYPE \
					--review-option $REVIEW_OPTION \
					--num-epoch 5
