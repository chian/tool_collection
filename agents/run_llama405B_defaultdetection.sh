for file in ~/Documents/ANL/ARC/chia/reports/txts/mm*.txt; do
    cmd="python read_mmwr_llama405B.py --input ${file} >> llama405B_defaultdetection3.txt"
    eval $cmd
done
