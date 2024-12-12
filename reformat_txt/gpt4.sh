#!/bin/bash

# Check if a prompt is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<prompt>\""
    exit 1
fi

# Use the first argument as the prompt
PROMPT=$1

curl -X POST https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/ \
     -H "Content-Type: application/json" \
     -d @- <<EOF | jq -r '.response'
{
  "user": "chia",
  "model": "gpt4",
  "system": "You are a helpful AI assistant.",
  "prompt": ["$PROMPT"],
  "temperature": 1.0,
  "logprobs": true,
  "top_logprobs": 2,
  "max_tokens": 1
}
EOF

# Check if curl command was successful
if [ $? -ne 0 ]; then
    echo "Failed"
    exit 1
fi

#  "logprobs": True,
#  "top_logprobs": 5,
#  "n": 5,
#  "max_tokens": 1
