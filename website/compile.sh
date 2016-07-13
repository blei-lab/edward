#!/bin/bash
rm -f *.html

cd tex
for filename in *.tex; do
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes \
         --to=html \
         --mathjax \
         --no-highlight \
         --title-prefix=Edward \
         --template=edward_template.pandoc \
         --output=../${filename%.*}.html
done

cd ..
python strip_p_in_li.py 

cd ../sphinx
sphinx-apidoc -f -e -M -T -o source/ ../edward
make html
cp -r build/html/* ../website/api

cd ../website
