#!/bin/bash

for L in 12 16 20 24 
do
    for b in $(seq 0.3 0.001 0.4)
    do
        julia main.jl $L -n 1000000 -s 10 --beta $b
    done
done