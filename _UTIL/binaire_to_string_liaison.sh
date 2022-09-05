#!/bin/bash
corpus="/mnt/nas/Users/Marion_O/Bench/Bench2022/BDS_dir/Bench_svi/BDS_4609/"
FILES="${corpus}BDS/dbg/*dbg"
# mkdir "${corpus}BDS/dbg_utf8"
# mkdir "${corpus}BDS/rule"
# mkdir "${corpus}BDS/dbg_l"

for filename in $(ls $FILES)
#for filename in $(ls -1 $FILES)
do
  echo $filename
  newname= printf '%s\n' "$filename" | sed -e "s/dbg/rule/g"  
  newnamedbg= printf '%s\n' "$filename" | sed -e "s/dbg/dbg_l/g"
  newname_utf= printf '%s\n' "$filename" | sed -e "s/dbg/dbg_utf8/g"
  # echo $newname
  # echo $newnamedbg
  # echo $newname_utf
  find .dbg -type f -exec iconv -f CP1252 -t UTF-8//TRANSLIT "{}" -o ${corpus}BDS/dbg_utf8/"{}" \;
  # iconv -f CP1252 -t UTF-8//TRANSLIT --output=${newname_utf} ${filename}
  grep -h -E "(Liaison Rule \(line ).*$" ${newname_utf} > ${newname}
  grep -h -E "(Liaison Rule .*)|(fr_liaiso.*LNGST\s+word).*$" ${newname_utf} > ${newnamedbg}

done;


