#!/bin/bash
# Example script setting up the rnv variables needed for running ConceptGraphs
# Please adapt it to your own paths!

export CG_FOLDER=~/PycharmProjects/concept-graphs

export GSA_PATH=/path/to/Grounded-Segment-Anything

export REPLICA_ROOT=~/PycharmProjects/concept-graphs/Datasets/Replica
export REPLICA_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/replica.yaml

#export OPENAI_API_KEY=<your GPT-4 API KEY here>

export LLAVA_PYTHON_PATH=/path/to/llava
export LLAVA_CKPT_PATH=/path/to/llava/checkpoint/folder