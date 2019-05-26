#!/bin/bash

echo "Enter experiment name : "
read OUTPUT_DIR

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

PERMISSION_TYPE="READ_CONTACTS"

TRAIN_TYPE="acnet"
TEST_TYPE="whyper"
PREVECTOR_TYPE="fasttext"


python runner.py 	--permission-type $PERMISSION_TYPE \
					--train /home/huseyinalecakir/Security/data/ac-net/ACNET_DATASET.csv \
					--train-type  $TRAIN_TYPE \
					--test /home/huseyinalecakir/Security/data/whyper/Read_Contacts.csv \
					--test-type $TEST_TYPE \
					--prevectors  /home/huseyinalecakir/Security/data/cc.en.300.bin \
					--prevectype $PREVECTOR_TYPE \
					--saved-parameters-dir  /home/huseyinalecakir/Security/data/saved_parameters \
					--saved-prevectors $PREVECTOR_TYPE-$PERMISSION_TYPE-embeddings.pickle \
                    --saved-vocab-train $TRAIN_TYPE-vocab-$PERMISSION_TYPE.txt \
                    --saved-vocab-test $TEST_TYPE-vocab-$PERMISSION_TYPE.txt \
					--outdir $OUTPUT_DIR



touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : \n\n" > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
