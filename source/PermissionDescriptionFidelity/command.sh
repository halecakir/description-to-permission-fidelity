#!/bin/bash

PERMISSION_TYPE=$1
EMBEDDING=$2
PREVECTOR_TYPE=$3

TRAIN_TYPE="acnet"
TEST_TYPE="whyper"
OUTPUT_DIR=./test/$PERMISSION_TYPE/$PREVECTOR_TYPE

# Create output directory if not exists
mkdir -p $OUTPUT_DIR
mkdir -p /home/huseyinalecakir/Security/data/saved_parameters/$PERMISSION_TYPE/$PREVECTOR_TYPE

python runner.py 	--permission-type $PERMISSION_TYPE \
					--train /home/huseyinalecakir/Security/data/ac-net/ACNET_DATASET.csv \
					--train-type  $TRAIN_TYPE \
					--test /home/huseyinalecakir/Security/data/whyper/Read_Contacts.csv \
					--test-type $TEST_TYPE \
					--prevectors  /home/huseyinalecakir/Security/data/$EMBEDDING \
					--prevectype $PREVECTOR_TYPE \
					--saved-parameters-dir  /home/huseyinalecakir/Security/data/saved_parameters/$PERMISSION_TYPE/$PREVECTOR_TYPE \
					--saved-prevectors embeddings.pickle \
                    --saved-vocab-train $TRAIN_TYPE-vocab.txt \
                    --saved-vocab-test $TEST_TYPE-vocab.txt \
                    --saved-sentences-whyper  $TEST_TYPE-preprocessed.txt \
                    --saved-sentences-acnet  $TRAIN_TYPE-preprocessed.txt \
					--outdir $OUTPUT_DIR



touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : " > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
