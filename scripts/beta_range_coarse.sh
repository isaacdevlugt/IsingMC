#!/bin/bash

for L in 12 16 20 24 
do
    for b in 0.05 0.1 0.2 1.0 2.0 5.0 10.0
    do
        julia main.jl $L -n 1000000 -s 10 --beta $b
    done
done