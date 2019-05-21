#!/bin/bash
set -eux
for e in Humanoid Ant Reacher Walker2d Hopper HalfCheetah    #Ant Humanoid
do
    pythonw DAgger.py experts/$e-v1.pkl $e-v2 --num_rollouts=5 #mac #--render
    #python DAgger.py experts/$e-v1.pkl $e-v2 --num_rollouts=5 
done
