#!/bin/bash

batch_size=32
epoch=500

#python main.py --batch_size $batch_size --data /data/awd-lstm-lm/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1 --save ~/tmp/PTB_debug.pt
python main.py --batch_size $batch_size --data /data/awd-lstm-lm/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch $epoch --save ~/tmp/PTB.pt
