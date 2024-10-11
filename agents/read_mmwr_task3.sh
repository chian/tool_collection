#Bash script stringing together all commands
OUTPUT_DIR='../../ARC/chia/reports/txts/output_claude3_task3'
INPUT_DIR='../../ARC/chia/reports/txts'
INPUT_FILE=$1 #e.g. mm6643-H.txt
python read_mmwr_claude3v3_dumber.py --input $INPUT_DIR/$INPUT_FILE > $OUTPUT_DIR/task3_$INPUT_FILE
for DATASET in "LOCATION" "TRANSMISSION" "INCUBATION_PERIOD" "INFECTIOUS_PERIOD" "PROBABILITY_TRANSMISSION" "INITIAL_INFECTED"; do
    cp dict_files/$DATASET.txt ../pull_papers/cmsc35350/$DATASET/.
    cd ../pull_papers/cmsc35350
    python source/bulk_fetch_abstracts_from_SS_via_keyword.py -d $DATASET -f $DATASET/${DATASET}.txt
    sort -u $DATASET/$DATASET.jsonl > $DATASET/${DATASET}_uniq.jsonl
    mv $DATASET/${DATASET}_uniq.jsonl $DATASET/${DATASET}.jsonl
    cd ../../agents
    python read_mmwr_claude3v3part2.py --input ../pull_papers/cmsc35350/$DATASET/$DATASET.jsonl --entity $DATASET >> $OUTPUT_DIR/task3_$INPUT_FILE
done
