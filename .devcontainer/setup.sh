#!/usr/bin/env bash
set -e

# install dependencies
pip install -r requirements.txt

# install as editable project
pip install --editable .
