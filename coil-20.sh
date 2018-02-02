#!/usr/bin/env bash
# This script splits the coil-20 <http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php>
# dataset into a random train and test dataset in the format for my dataset generator

IN_DIR=$1

traindir=$IN_DIR/train
testdir=$IN_DIR/test


# split them into the right folder
cd $IN_DIR
mkdir train
mkdir test
for i in `seq 1 20`
do

    mkdir train/$i
    mkdir test/$i
    n="obj$i"
    n+="_*"
    mv $n train/$i

    # get all those files in that directory, shuffle them, and put the first n in the test directory
    numimages=$(ls train/$i | wc -l)
    num_test=$[$numimages / 6]
    echo "$num_test"
    for f in $(ls train/$i | shuf -n $num_test)
    do
        mv train/$i/$f test/$i
    done
done









