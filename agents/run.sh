for file in ~/Documents/ANL/ARC/chia/reports/txts/mm670*.txt; do
    filename=$(basename "$file")
    output_file="~/Documents/ANL/ARC/chia/reports/txts/output_claude3_task2/${filename%.txt}.task2.txt"
    cmd="python read_mmwr_claude3v2.py --input ${file} > ${output_file}"
    echo $cmd
    eval $cmd
    echo "Processed $filename"
    sleep 60
done
