#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo $OSTYPE
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    name_script="linux_sim"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    name_script="mac_sim"
else
    echo "OS not recognized"
fi

base_url="http://virtual-home.org/release/simulator/"
url=$base_url$name_script".zip"
echo "Downloading"
wget $url -P "$PROJECT_ROOT/simulation/"
cd "$PROJECT_ROOT/simulation/"
unzip $name_script.zip
rm $name_script.zip
cd "$PROJECT_ROOT"
echo "Executable moved to simulation folder"
k