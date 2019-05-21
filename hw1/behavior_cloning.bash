#!/bin/bash
set -eux
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
do
    pythonw behavior_cloning.py experts/$e-v1.pkl $e-v2 --num_rollouts=5 #mac
    #python behavior_cloning.py experts/$e-v1.pkl $e-v2 --num_rollouts=5 
done
