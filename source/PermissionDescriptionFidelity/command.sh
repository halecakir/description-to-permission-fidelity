#!/bin/bash

ROOT_DIR=$1
PERMISSION_TYPE=$2
EMBEDDING=$3
PREVECTOR_TYPE=$4
STEMMER=$5
EMBEDDING_DIM=$6
LSTM_TYPE=$7
EXTERNAL_INFO=$8
EXTERNAL_INFO_DIM=$9

TRAIN_TYPE="acnet"
TEST_TYPE="whyper"
OUTPUT_DIR=./$ROOT_DIR/$PERMISSION_TYPE/$PREVECTOR_TYPE/$STEMMER/$EMBEDDING_DIM

# Create output directory if not exists
mkdir -p $OUTPUT_DIR
mkdir -p /home/huseyinalecakir/Security/data/saved_parameters/$PERMISSION_TYPE/$PREVECTOR_TYPE/$STEMMER/$EMBEDDING_DIM

python runner.py 	--permission-type $PERMISSION_TYPE \
					--train /home/huseyinalecakir/Security/data/ac-net/ACNET_DATASET.csv \
					--train-type  $TRAIN_TYPE \
					--test /home/huseyinalecakir/Security/data/whyper/Read_Contacts.csv \
					--test-type $TEST_TYPE \
					--prevectors  /home/huseyinalecakir/Security/data/$EMBEDDING \
					--prevectype $PREVECTOR_TYPE \
					--saved-parameters-dir  /home/huseyinalecakir/Security/data/saved_parameters/$PERMISSION_TYPE/$PREVECTOR_TYPE/$STEMMER/$EMBEDDING_DIM  \
					--saved-prevectors embeddings.pickle \
					--saved-vocab-train $TRAIN_TYPE-vocab.txt \
					--saved-vocab-test $TEST_TYPE-vocab.txt \
					--saved-sentences-whyper  $TEST_TYPE-preprocessed.txt \
					--saved-sentences-acnet  $TRAIN_TYPE-preprocessed.txt \
					--outdir $OUTPUT_DIR \
					--stemmer $STEMMER \
					--wembedding $EMBEDDING_DIM \
					--lstm-type $LSTM_TYPE \
					--external-info $EXTERNAL_INFO \
					--external-info-dim $EXTERNAL_INFO_DIM




touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : " > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
