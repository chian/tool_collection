#!/bin/bash

# Check if an input (directory or file) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory or file>"
    exit 1
fi

input_path="$1"
chunk_size=2000 # Number of words per chunk

process_chunk() {
    local chunk_text="$1"
    local json_text=$(jq -aRs . <<< "$chunk_text")

    # Send the request to summarize the chunk
    local summary=$(curl -s -X POST https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/ \
        -H "Content-Type: application/json" \
        -d "{\"user\":\"mtdapi\", \"model\":\"gpt4\", \"system\": \"You are a helpful operations assistant. You specialize in understanding summarizing scientific papers.\", \"prompt\":[$json_text], \"stop\":[], \"temperature\":0.8, \"top_p\":0.7}")

    echo "$summary"
}

process_file() {
    local file=$1
    echo "Processing $file..."
    
    local cat_summary=""
    local file_content=$(cat "$file")
    #echo "$file_content"
    local word_array=($file_content)
    #for word in "${word_array[@]}"; do
    #   echo "$word"
    #done
    local total_words=${#word_array[@]}
    echo "Total words: $total_words"
    local current_word=0

    while [ $current_word -lt $total_words ]; do
        local chunk=("${word_array[@]:$current_word:$chunk_size}")
	#echo "$chunk"
        local chunk_text="${chunk[*]}"
	#echo "$chunk_text"
        local summary=$(process_chunk "$chunk_text")
        cat_summary+="$summary "
        let current_word+=chunk_size
    done

    echo "Final Summary for $file:"
    local full_summary=$(curl -s -X POST https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/ \
        -H "Content-Type: application/json" \
        -d "{\"user\":\"mtdapi\", \"model\":\"gpt4\", \"system\": \"You are a helpful operations assistant. You specialize in understanding summarizing scientific papers.\", \"prompt\":[\"Summarize this text: $cat_summary\"], \"stop\":[], \"temperature\":0.8, \"top_p\":0.7}")
    echo "$full_summary"
    echo
}

if [ -d "$input_path" ]; then
    # It's a directory, process each file
    for file in "$input_path"/*; do
        process_file "$file"
    done
elif [ -f "$input_path" ]; then
    # It's a file, process the file
    process_file "$input_path"
else
    echo "The input path is not a valid directory or file"
    exit 2
fi
