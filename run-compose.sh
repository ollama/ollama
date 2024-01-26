#!/bin/bash

# Define color and formatting codes
BOLD='\033[1m'
GREEN='\033[1;32m'
WHITE='\033[1;37m'
RED='\033[0;31m'
NC='\033[0m' # No Color
# Unicode character for tick mark
TICK='\u2713'

# Detect GPU driver
get_gpu_driver() {
    # Detect NVIDIA GPUs using lspci or nvidia-smi
    if lspci | grep -i nvidia >/dev/null || nvidia-smi >/dev/null 2>&1; then
        echo "nvidia"
        return
    fi

    # Detect AMD GPUs (including GCN architecture check for amdgpu vs radeon)
    if lspci | grep -i amd >/dev/null; then
        # List of known GCN and later architecture cards
        # This is a simplified list, and in a real-world scenario, you'd want a more comprehensive one
        local gcn_and_later=("Radeon HD 7000" "Radeon HD 8000" "Radeon R5" "Radeon R7" "Radeon R9" "Radeon RX")

        # Get GPU information
        local gpu_info=$(lspci | grep -i 'vga.*amd')

        for model in "${gcn_and_later[@]}"; do
            if echo "$gpu_info" | grep -iq "$model"; then
                echo "amdgpu"
                return
            fi
        done

        # Default to radeon if no GCN or later architecture is detected
        echo "radeon"
        return
    fi

    # Detect Intel GPUs
    if lspci | grep -i intel >/dev/null; then
        echo "i915"
        return
    fi

    # If no known GPU is detected
    echo "Unknown or unsupported GPU driver"
    exit 1
}

# Function for rolling animation
show_loading() {
    local spin='-\|/'
    local i=0

    printf " "

    while kill -0 $1 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\b${spin:$i:1}"
        sleep .1
    done

    # Replace the spinner with a tick
    printf "\b${GREEN}${TICK}${NC}"
}

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --enable-gpu[count=COUNT]  Enable GPU support with the specified count."
    echo "  --enable-api[port=PORT]    Enable API and expose it on the specified port."
    echo "  --webui[port=PORT]         Set the port for the web user interface."
    echo "  --data[folder=PATH]        Bind mount for ollama data folder (by default will create the 'ollama' volume)."
    echo "  --build                    Build the docker image before running the compose project."
    echo "  --drop                     Drop the compose project."
    echo "  -q, --quiet                Run script in headless mode."
    echo "  -h, --help                 Show this help message."
    echo ""
    echo "Examples:"
    echo "  $0 --drop"
    echo "  $0 --enable-gpu[count=1]"
    echo "  $0 --enable-api[port=11435]"
    echo "  $0 --enable-gpu[count=1] --enable-api[port=12345] --webui[port=3000]"
    echo "  $0 --enable-gpu[count=1] --enable-api[port=12345] --webui[port=3000] --data[folder=./ollama-data]"
    echo "  $0 --enable-gpu[count=1] --enable-api[port=12345] --webui[port=3000] --data[folder=./ollama-data] --build"
    echo ""
    echo "This script configures and runs a docker-compose setup with optional GPU support, API exposure, and web UI configuration."
    echo "About the gpu to use, the script automatically detects it using the "lspci" command."
    echo "In this case the gpu detected is: $(get_gpu_driver)"
}

# Default values
gpu_count=1
api_port=11435
webui_port=3000
headless=false
build_image=false
kill_compose=false

# Function to extract value from the parameter
extract_value() {
    echo "$1" | sed -E 's/.*\[.*=(.*)\].*/\1/; t; s/.*//'
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --enable-gpu*)
            enable_gpu=true
            value=$(extract_value "$key")
            gpu_count=${value:-1}
            ;;
        --enable-api*)
            enable_api=true
            value=$(extract_value "$key")
            api_port=${value:-11435}
            ;;
        --webui*)
            value=$(extract_value "$key")
            webui_port=${value:-3000}
            ;;
        --data*)
            value=$(extract_value "$key")
            data_dir=${value:-"./ollama-data"}
            ;;
        --drop)
            kill_compose=true
            ;;
        --build)
            build_image=true
            ;;
        -q|--quiet)
            headless=true
            ;;
        -h|--help)
            usage
            exit
            ;;
        *)
            # Unknown option
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift # past argument or value
done

if [[ $kill_compose == true ]]; then
    docker compose down --remove-orphans
    echo -e "${GREEN}${BOLD}Compose project dropped successfully.${NC}"
    exit
else
    DEFAULT_COMPOSE_COMMAND="docker compose -f docker-compose.yaml"
    if [[ $enable_gpu == true ]]; then
        # Validate and process command-line arguments
        if [[ -n $gpu_count ]]; then
            if ! [[ $gpu_count =~ ^[0-9]+$ ]]; then
                echo "Invalid GPU count: $gpu_count"
                exit 1
            fi
            echo "Enabling GPU with $gpu_count GPUs"
            # Add your GPU allocation logic here
            export OLLAMA_GPU_DRIVER=$(get_gpu_driver)
            export OLLAMA_GPU_COUNT=$gpu_count # Set OLLAMA_GPU_COUNT environment variable
        fi
        DEFAULT_COMPOSE_COMMAND+=" -f docker-compose.gpu.yaml"
    fi
    if [[ $enable_api == true ]]; then
        DEFAULT_COMPOSE_COMMAND+=" -f docker-compose.api.yaml"
        if [[ -n $api_port ]]; then
            export OLLAMA_WEBAPI_PORT=$api_port # Set OLLAMA_WEBAPI_PORT environment variable
        fi
    fi
    if [[ -n $data_dir ]]; then
        DEFAULT_COMPOSE_COMMAND+=" -f docker-compose.data.yaml"
        export OLLAMA_DATA_DIR=$data_dir # Set OLLAMA_DATA_DIR environment variable
    fi
    if [[ -n $webui_port ]]; then
        export OLLAMA_WEBUI_PORT=$webui_port # Set OLLAMA_WEBUI_PORT environment variable
    fi
    DEFAULT_COMPOSE_COMMAND+=" up -d"
    DEFAULT_COMPOSE_COMMAND+=" --remove-orphans"
    DEFAULT_COMPOSE_COMMAND+=" --force-recreate"
    if [[ $build_image == true ]]; then
        DEFAULT_COMPOSE_COMMAND+=" --build"
    fi
fi

# Recap of environment variables
echo
echo -e "${WHITE}${BOLD}Current Setup:${NC}"
echo -e "   ${GREEN}${BOLD}GPU Driver:${NC} ${OLLAMA_GPU_DRIVER:-Not Enabled}"
echo -e "   ${GREEN}${BOLD}GPU Count:${NC} ${OLLAMA_GPU_COUNT:-Not Enabled}"
echo -e "   ${GREEN}${BOLD}WebAPI Port:${NC} ${OLLAMA_WEBAPI_PORT:-Not Enabled}"
echo -e "   ${GREEN}${BOLD}Data Folder:${NC} ${data_dir:-Using ollama volume}"
echo -e "   ${GREEN}${BOLD}WebUI Port:${NC} $webui_port"
echo

if [[ $headless == true ]]; then
    echo -ne "${WHITE}${BOLD}Running in headless mode... ${NC}"
    choice="y"
else
    # Ask for user acceptance
    echo -ne "${WHITE}${BOLD}Do you want to proceed with current setup? (Y/n): ${NC}"
    read -n1 -s choice
fi

echo

if [[ $choice == "" || $choice == "y" ]]; then
    # Execute the command with the current user
    eval "$DEFAULT_COMPOSE_COMMAND" &

    # Capture the background process PID
    PID=$!

    # Display the loading animation
    #show_loading $PID

    # Wait for the command to finish
    wait $PID

    echo
    # Check exit status
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}${BOLD}Compose project started successfully.${NC}"
    else
        echo -e "${RED}${BOLD}There was an error starting the compose project.${NC}"
    fi
else
    echo "Aborted."
fi

echo
