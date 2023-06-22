#!/bin/bash

# Function to handle Ctrl+C
handle_sigint() {
    kill $pid1 $pid2 $pid3
    exit
}

# Trap Ctrl+C signal
trap 'handle_sigint' SIGINT

# Start three processes in the background
npm run dev --prefix ./client & pid1=$!
npm start --prefix ./desktop & pid2=$!
go run -C ./server . & pid3=$!

# Wait for all processes to finish
wait
