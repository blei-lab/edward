#!/bin/bash
rm -f *.html
rm -rf api

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

# Strip paragraphs in lists in pandoc's html output
cd ..
python strip_p_in_li.py

# Run sphinx to generate the API
sphinx-apidoc -f -e -M -T -o source/ ../edward
make html
mkdir -p api
cp -r build/html/* api
