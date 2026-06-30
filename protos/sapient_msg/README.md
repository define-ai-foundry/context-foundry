# SAPIENT (BSI Flex 335) Protocol Definitions

This directory contains the Protocol Buffer (.proto) definitions for the SAPIENT BSI Flex 335 specification.

## Provenance
These files are sourced from the official Dstl (Defence Science and Technology Laboratory) repository:
- **Repository:** [Dstl/sapient-framework](https://github.com/dstl/sapient-framework) 
  *(Replace with your specific internal/external Dstl repo URL)*

## Purpose
These definitions form the "Gatekeeper" of the fusion engine. All binary data received from SAPIENT sensors must adhere to these schemas. 

## Compilation
To regenerate the Python bindings after updating these files, run the following from the repository root:
```bash
python -m grpc_tools.protoc --proto_path=protos --python_out=src protos/sapient_msg/bsi_flex_335_v2_0/*.proto