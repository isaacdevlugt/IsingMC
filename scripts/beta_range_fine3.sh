#!/bin/bash

for L in 12 16 20 24 
do
    for b in $(seq 0.501 0.001 0.6)
    do
        julia main.jl $L -n 1000000 --beta $b --seed 9999
    done
done
