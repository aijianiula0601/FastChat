#!/bin/bash

set -ex

cd ../

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

python fastchat/train/test_dataset.py