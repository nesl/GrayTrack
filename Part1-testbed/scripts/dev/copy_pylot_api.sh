#!/bin/bash

mkdir /home/ubuntu/PYLOT_API

pip install networkx==2.2

docker cp slow-pylot:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/carla /home/ubuntu/PYLOT_API

echo "add the following to .bashrc"

echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PYLOT_API/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PYLOT_API/PythonAPI/carla/agents/navigation/"
echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PYLOT_API/PythonAPI/carla"
echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/PYLOT_API/PythonAPI"

echo "source .bashrc"
