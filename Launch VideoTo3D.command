#!/bin/bash
cd "/Users/mohamadjlo/Desktop/VideoTo3D"

# Check if port 5050 is already running
if ! lsof -i :5050 > /dev/null 2>&1; then
    echo "Starting Video-to-3D Local Server..."
    nohup /Users/mohamadjlo/Desktop/.3dgs_toolchain/venv_3dgs/bin/python server.py 5050 > server.log 2>&1 &
    sleep 1.5
fi

# Open in default browser
open "http://127.0.0.1:5050"
