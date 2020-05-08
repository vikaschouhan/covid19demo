#!/bin/bash

function usage {
    echo "$0 config_file"
    exit
}

# If no args passed, display usage
if [ $# -lt 1 ]; then
    usage
fi

# Vars
config_file=$1
export PYTHONPATH=.

# Run in loop
while true
do
    run_cmd="python3 server.py --config_file $config_file"
    echo "CMD : $run_cmd"
    eval $run_cmd
done
