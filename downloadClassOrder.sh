#!/usr/bin/env bash


declare -a orders=("sub/rebekah.esm/55400")
savloc=/home/rebekah/stability/data
dateval=$(date +'%Y%m%d')

find $savloc -type f -mtime +3 -exec rm {} \;

cd ${savloc}

for order in "${orders[@]}"; do
  wget --no-clobber ftp://ftp.avl.class.noaa.gov/${order}/*s${dateval}* .
done

for f in *.tar; do
 tar -xvf $f
done

rm ${savloc}/*.tar

# cd -
