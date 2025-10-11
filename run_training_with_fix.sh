#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
source tsl-env/bin/activate
python3 scripts/train/train_celestial_direct.py
