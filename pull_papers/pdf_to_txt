#!/bin/bash

# Check if a directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory containing the PDF files
directory="$1"

# Loop through each PDF file in the directory
for pdf_file in "$directory"/*.pdf; do
    # Skip if the file is not a PDF
    if [[ ! $pdf_file == *.pdf ]]; then
        continue
    fi

    # Construct the TXT filename by replacing the PDF extension
    txt_file="${pdf_file%.pdf}.txt"

    # Check if the TXT file already exists, skip if it does
    if [ -f "$txt_file" ]; then
        echo "Skipping existing file: $txt_file"
        continue
    fi

    # Use curl to process the PDF and save the output to the TXT file
    curl -X POST https://apps-dev.inside.anl.gov/gottextai/api/v1/extracttext \
         -F "file=@$pdf_file" \
         -F "clean_for_corpus=False" \
         -F "simple_clean_text=True" \
         -F "simple_summary=False" > "$txt_file"

    # Check if curl command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to process $pdf_file"
        continue
    fi

    echo "Processed and saved: $txt_file"
done

echo "Processing complete."
