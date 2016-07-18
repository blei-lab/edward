#!/bin/bash
rm -f *.html

# Compile all the tex files into html
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

# Compile index.tex into markdown, to make it easier to copy
# to the base level README.md
pandoc index.tex --output=../index.md

# Strip paragraphs in lists in pandoc's html output
cd ..
python strip_p_in_li.py

# Run sphinx to generate the API
cd ../docs
sphinx-apidoc -f -e -M -T -o source/ ../edward
make html
cp -r build/html/* ../website/api

cd ../website
