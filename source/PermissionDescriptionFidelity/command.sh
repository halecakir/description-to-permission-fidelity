#!/bin/bash

echo "Enter experiment name : "
read OUTPUT_DIR

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

python runner.py --train /home/huseyinalecakir/Security/data/whyper/Read_Calendar.xls \
		 --train-type excel \
		 --prevectors  /home/huseyinalecakir/Security/data/cc.en.300.bin \
		 --prevectype fasttext \
		 --saved-parameters-dir  /home/huseyinalecakir/Security/data/saved_parameters \
		 --saved-prevectors embeddings.pickle \
		 --saved-vocab vocab.txt \
		 --outdir $OUTPUT_DIR 


touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : \n\n" > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
