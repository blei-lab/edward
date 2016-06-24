#!/bin/bash
rm -f *.html

cd tex
for filename in *.tex; do
  pandoc ${filename%.*}.tex \
         --mathjax \
         --no-highlight \
         --title-prefix=Edward \
         --template=edward_template.pandoc \
         -o ../${filename%.*}.html
done

cd ..
python strip_p_in_li.py 

cd ../sphinx
make html
cp -r build/html/* ../website/api
