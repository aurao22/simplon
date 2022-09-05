#!/bin/bash
for file in *.txt; do
     mv "$file" "$(sed 's/e0/e/g' <<< "$file")"
done