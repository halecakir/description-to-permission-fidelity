#!/bin/bash

echo "Enter experiment name : "
read OUTPUT_DIR

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

python runner.py 	--train /home/huseyinalecakir/Security/data/small_processed/apps_mini_processed.csv \
					--train-type  csv \
					--test /home/huseyinalecakir/Security/data/whyper/Read_Calendar.xls \
					--test-type excel \
					--prevectors  /home/huseyinalecakir/Security/data/cc.en.300.bin \
					--prevectype fasttext \
					--saved-parameters-dir  /home/huseyinalecakir/Security/data/saved_parameters \
					--saved-prevectors fasttext-read-calendar-embeddings.pickle
                    --saved-vocab-train, raw-vocab-read-calendar.txt
                    --saved-vocab-test whyper-vocab-read-calendar.txt
					--outdir $OUTPUT_DIR


touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : \n\n" > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
