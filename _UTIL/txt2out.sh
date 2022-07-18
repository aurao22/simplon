#!/bin/bash
corpus=$1
FILES="${corpus}BDS/txt/*txt"

mkdir "${corpus}BDS/dbg"
mkdir "${corpus}BDS/out"
for filename in $(ls $FILES)
do
  newname=printf '%s\n' "$filename" | sed -e "s/txt/out/g"
  newnamedbg=printf '%s\n' "$filename" | sed -e "s/txt/dbg/g"
  echo $newname
  ./baratinoo -g 5  -l fr-FR  -v Videfr -i utf8 $filename -o bintext $newname ./baratinoo.cfg > $newnamedbg 2>&1
done;
# csplit -f phrase -n 5 -b  “%05d.txt” -n 5  liste_liaisons_interdites.txt 1  {207}

