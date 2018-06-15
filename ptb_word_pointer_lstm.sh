#!/bin/bash

python pointer.py --data /data/awd-lstm-lm/penn --save ~/tmp/PTB.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000
