#!/bin/bash
# Add these lines to your ~/.bashrc file

echo "# AMD GPU ROCm fixes for Rembrandt (gfx1151)" >> ~/.bashrc
echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" >> ~/.bashrc
echo "export AMD_SERIALIZE_KERNEL=1" >> ~/.bashrc
echo "export HIP_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export ROCR_VISIBLE_DEVICES=0" >> ~/.bashrc

echo "Environment variables added to ~/.bashrc"
echo "Run 'source ~/.bashrc' or restart your terminal"