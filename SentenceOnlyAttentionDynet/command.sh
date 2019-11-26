#!/bin/bash

PERMISSION_TYPE=$1
EXPERIMENT_TYPE=$2
ENCODER_DIR=$3
ENCODER_TYPE=$4



MODEL_TYPE="SentenceOnlyAttentionDyNET"
OUTPUT_DIR="../output/$MODEL_TYPE"
PARAMETERS_DIR="/home/huseyinalecakir/Security/data/saved-parameters"

# Create output directory if not exists
mkdir -p $OUTPUT_DIR
OUT_FILE=$OUTPUT_DIR/$PERMISSION_TYPE-$EXPERIMENT_TYPE.out
rm -f $OUT_FILE

python runner.py 	--permission-type $PERMISSION_TYPE \
					--saved-data $PARAMETERS_DIR/saved-data/emdeddings-sentences-w2i.pickle \
					--saved-reviews $PARAMETERS_DIR/saved-data/reviews.pickle \
					--saved-predicted-reviews $PARAMETERS_DIR/saved-data/predicted-$PERMISSION_TYPE-reviews.pickle \
					--model-checkpoint $PARAMETERS_DIR/saved-models/$MODEL_TYPE-$PERMISSION_TYPE.pt \
					--outdir $OUT_FILE \
					--encoder-dir $3 \
					--encoder-type $4

