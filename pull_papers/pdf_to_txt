#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <PDF file or directory>"
  exit 1
fi

# Assign the first argument to a variable
PDF_INPUT="$1"

# Determine the directory of the PDF input and process accordingly
if [ -d "$PDF_INPUT" ]; then
  # If the input is a directory, process all PDF files in it
  for PDF_FILE in "$PDF_INPUT"/*.pdf; do
    PDF_NAME=$(basename "$PDF_FILE")
    TXT_NAME="${PDF_NAME%.pdf}.txt"
    ../pull_papers/nougat_pdf.py "$PDF_FILE"
    if [ -f "$PDF_INPUT/$TXT_NAME" ]; then
      echo "Processing $PDF_INPUT/$TXT_NAME with read_mmwr.py"
      ./read_mmwr.py --input "$PDF_INPUT/$TXT_NAME" --paragraph
    else
      echo "No output file for $PDF_FILE found in $PDF_INPUT"
    fi
  done
elif [ -f "$PDF_INPUT" ]; then
  # If the input is a file, process this file only
  PDF_DIR=$(dirname "$PDF_INPUT")
  PDF_NAME=$(basename "$PDF_INPUT")
  TXT_NAME="${PDF_NAME%.pdf}.txt"
  ../pull_papers/nougat_pdf.py "$PDF_INPUT"
else
  echo "The specified PDF file or directory does not exist."
  exit 1
fi
