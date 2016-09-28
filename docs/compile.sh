#!/bin/bash
echo "Compiling website."

echo "Clearing all html files."
rm -f *.html
rm -rf tutorials
rm -rf api

# Compile all the tex files into html
echo "Begin pandoc compilation."
mkdir -p tutorials
cd tex
for filename in {./,tutorials/}*.tex; do
  echo $filename
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes \
         --to=html \
         --mathjax \
         --no-highlight \
         --bibliography=bib.bib \
         --csl=apa.csl \
         --title-prefix=Edward \
         --template=template.pandoc \
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
