#!/bin/bash
rm out*.log
echo -e "\nnew run" >> out.log
date >> out.log

for fname in zeros rnd1mb image.pgm
do
    for mode in 0 1 15
    do
        for iter in 1 10 100
        do
            pcmd="./a.out $fname 1024 1024 $mode $iter"
            echo "timing $pcmd"
            $pcmd 2>&1 | tee -a out.log
        done
    done
done