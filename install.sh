#!/bin/bash
set -e

# Clone & install pytorch3d
echo "INSTALLING PYTORCH3d"
cd extensions
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python setup.py install --user
cd ..

# Build and install the cnms extension
echo "INSTALLING CNMS"
cd ../cnms
python setup.py install --user
cd ../..