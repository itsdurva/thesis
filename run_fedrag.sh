#!/bin/bash
set -e
shopt -s nullglob

for config in configs/*.toml; do
    echo " Using config: $config"
    cp -p "$config" pyproject.toml
    echo " Installing dependencies for config: $config at $(date) "
    pip install -e .
    echo " Starting run for config: $config at $(date) "
    flwr run .
    echo " Ending run for config: $config at $(date) "
done

rm "pyproject.toml"