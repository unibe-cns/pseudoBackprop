#!bin/bash
# load the current folder into the pythonpath
# to be able to execute the experiments from the project folders
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

echo "Load by init file"
