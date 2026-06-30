# Cursor on Target (CoT) Schemas

This directory contains the official MITRE Corporation Cursor on Target (CoT) XML schema definitions (XSD). 

## Provenance
These files are sourced from the Department of Defense (DoD) open-source repository:
- **Source:** [https://github.com/deptofdefense/AndroidTacticalAssaultKit-CIV](https://github.com/deptofdefense/AndroidTacticalAssaultKit-CIV)
- **Folder:** `takcot/mitre/`
- **Version:** 4.2.0.1 (#61)

## Purpose
These schemas are used by the `CotValidator` in the fusion engine to rigorously validate incoming CoT XML messages before they are processed by the tracker. Do not modify these files manually; update them by syncing with the official upstream repository.