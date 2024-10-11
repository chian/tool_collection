for file in ~/Documents/ANL/ARC/chia/reports/txts/mm*.txt; do
    cmd="python read_mmwr_llama405B_Aug16.py --input ${file} >> llama405B_15onlydetection_Aug16_2.txt"
    eval $cmd
done
