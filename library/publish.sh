#!/bin/bash

mkdir -p models

# download binaries
function process_line {
    local url=$1
    local checksum=$2

    # Get the filename from the URL
    local filename=models/$(basename $url)

    echo "verifying $filename..."

    # If the file exists, compute its checksum
    if [ -f $filename ]; then
        local existing_checksum=$(shasum -a 256 $filename | cut -d ' ' -f1)
    fi

    # If the file does not exist, or its checksum does not match, download it
    if [ ! -f $filename ] || [ $existing_checksum != $checksum ]; then
        echo "downloading $filename..."
        
        # Download the file
        curl -L $url -o $filename

        # Compute the SHA256 hash of the downloaded file
        local computed_checksum=$(shasum -a 256 $filename | cut -d ' ' -f1)

        # Verify the checksum
        if [ $computed_checksum != $checksum ]; then
            echo "Checksum verification failed for $filename"
            exit 1
        fi
    fi
}

while IFS=' ' read -r url checksum
do
    process_line $url $checksum
done < "downloads"

# create and publish the models
for file in modelfiles/*; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo $filename
    ollama create "library/${filename}" -f "$file"
    ollama push "${filename}"
  fi
done

