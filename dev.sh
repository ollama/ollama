#!/bin/bash

# Function to handle Ctrl+C
handle_sigint() {
    kill $pid1 $pid2
    exit
}

# Trap Ctrl+C signal
trap 'handle_sigint' SIGINT

# Start three processes in the background
npm run dev --prefix ./client & pid1=$!
npm start --prefix ./desktop & pid2=$!

# Wait for all processes to finish
wait
