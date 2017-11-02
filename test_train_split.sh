#!/usr/bin/env bash
IN_DIR=$1

traindir=$IN_DIR/train
testdir=$IN_DIR/test

rm -r $traindir
rm -r $testdir

numimages=$(ls $IN_DIR/*.png | wc -l)

mkdir $traindir
mkdir $testdir

num_test=$[$numimages / 6]

echo $num_test

shuffled=$(ls $IN_DIR/*.png | shuf)

train_files=("${shuffled[@]:0:$num_test}")
test_files=("${shuffled[@]:$num_test}")

echo "$($train_files | sort)"


for f in $($train_files | sort)
do
    cp $f $traindir
done

for f in $($test_files | sort)
do
    cp $f $testdir
done

