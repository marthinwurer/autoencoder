#!/bin/bash
# this file will be used to convert images to the correct format to be added to a dataset
IN_DIR=$1
OUT_DIR=$2

# la

CUR_NUM=`ls $OUT_DIR | sed -e 's/[^1-9]*\([0-9]*\)[^0-9]*/\1/' | sort -n | tail -1`
echo $CUR_NUM
CUR_NUM=$[$CUR_NUM + 1]
echo $CUR_NUM

for f in $(ls $IN_DIR )
do
#    echo $f $CUR_NUM
    convert -resize 128x64 $IN_DIR$f $OUT_DIR/out-$(printf "%05d" $CUR_NUM).png && CUR_NUM=$[$CUR_NUM + 1]
done
echo $CUR_NUM
