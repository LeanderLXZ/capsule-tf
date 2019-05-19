#!/usr/bin/env bash

if [ ! -d "../archived_logs" ]; then
    mkdir ../archived_logs
fi

if [ ! -d "../archived_logs/tf_logs" ]; then
    mkdir ../archived_logs/tf_logs
fi

if [ ! -d "../archived_logs/train_logs" ]; then
    mkdir ../archived_logs/train_logs
fi

if [ ! -d "../archived_logs/test_logs" ]; then
    mkdir ../archived_logs/test_logs
fi

if [ ! -d "../archived_logs/checkpoints" ]; then
    mkdir ../archived_logs/checkpoints
fi

mv ../tf_logs/* ../archived_logs/tf_logs/
mv ../train_logs/* ../archived_logs/train_logs/
mv  ../test_logs/* ../archived_logs/test_logs/
mv ../checkpoints/* ../archived_logs/checkpoints/
