#!/bin/bash -eu

echo "update pip"
pip install -U pip

echo "install submodules"
git submodule init
git submodule update --recursive

echo "install nmtpytorch"
pip install -e nmtpytorch
