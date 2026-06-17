#!/bin/bash
# Compiles vendored SAPIENT protobufs into the Python source directory

echo "Installing Python grpc-tools (compiler)..."
pip install grpcio-tools

echo "Compiling SAPIENT schemas..."
# OUTPUT DIRECTORY CHANGED TO ./src/
python -m grpc_tools.protoc \
    -I=./protos/ \
    --python_out=./src/ \
    ./protos/sapient_msg/bsi_flex_335_v2_0/*.proto \
    ./protos/sapient_msg/*.proto

# Create the __init__.py files in the NEW location
touch src/sapient_msg/__init__.py
touch src/sapient_msg/bsi_flex_335_v2_0/__init__.py

echo "Compilation complete! Python bindings generated in src/sapient_msg/"