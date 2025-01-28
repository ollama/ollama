#!/bin/bash

kill_process_tree() {
    local pid=$1
    # Get all child processes using pgrep
    local children=$(pgrep -P $pid)
    
    # Kill children first
    for child in $children; do
        kill_process_tree $child
    done
    
    # Kill the parent process
    kill -9 $pid 2>/dev/null || true
}

# Function to run the runner and benchmark for a given model
run_benchmark() {
    local model=$1
    
    echo "Starting runner with model: $model"
    # Start the runner in background and save its PID
    go run ../cmd/runner/main.go --new-runner -model "$model" &
    runner_pid=$!
    
    # Wait for the runner to initialize (adjust sleep time as needed)
    sleep 5
    
    echo "Running benchmark..."
    # Run test and wait for it to complete
    go test -bench=Runner
    test_exit_code=$?
    
    echo "Stopping runner process..."
    # Kill the runner process and all its children
    kill_process_tree $runner_pid
    
    # Wait for the process to fully terminate
    wait $runner_pid 2>/dev/null || true
    
    # Make sure no processes are still listening on port 8080
    lsof -t -i:8080 | xargs kill -9 2>/dev/null || true
    
    # Additional sleep to ensure port is freed
    sleep 2
    
    # Check if test failed
    if [ $test_exit_code -ne 0 ]; then
        echo "Warning: Benchmark test failed with exit code $test_exit_code"
    fi
    
    echo "Benchmark complete for model: $model"
    echo "----------------------------------------"
}


HOME_DIR="$HOME"
# llama3.2:1b:  ~/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45
# llama3.1:8b:  ~/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29
# llama3.3:70b: ~/.ollama/models/blobs/sha256-4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d
models=(
    "${HOME_DIR}/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    "${HOME_DIR}/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    # "${HOME_DIR}/.ollama/models/blobs/sha256-4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d"
)

# Run benchmarks for each model
for model in "${models[@]}"; do
    run_benchmark "$model"
done

echo "All benchmarks completed!"
