#!/bin/bash

python main.py --batch_size 20 --data /data/awd-lstm-lm/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1 --save ~/tmp/PTB_debug.pt
#python main.py --batch_size 20 --data /data/awd-lstm-lm/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save ~/tmp/PTB.pt
