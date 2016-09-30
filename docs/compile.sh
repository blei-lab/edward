#!/bin/bash
echo "Compiling website."

echo "Clearing all html files."
rm -f *.html
rm -rf api
rm -rf tutorials

# Compile all the tex files into html
echo "Begin pandoc compilation."
mkdir -p api
mkdir -p tutorials
cd tex
for filename in {./,api/,tutorials/}*.tex; do
  echo $filename
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter="../pandoc-code2raw.py" \
         --mathjax \
         --no-highlight \
         --bibliography=bib.bib \
         --csl=apa.csl \
         --title-prefix="Edward" \
         --template=template.pandoc \
         --output=../${filename%.*}.html
done

# Strip paragraphs in lists in pandoc's html output
cd ..
python strip_p_in_li.py

# # Generate docstrings
# python autogen.py

# # Compile all the API tex files (with generated docstrings) into html
# mkdir -p api
# cd build
# for filename in *.tex; do
#   pandoc ${filename%.*}.tex \
#          --from=latex+link_attributes \
#          --to=html \
#          --mathjax \
#          --no-highlight \
#          --bibliography=../tex/bib.bib \
#          --csl=../tex/apa.csl \
#          --title-prefix="Edward API" \
#          --template=../tex/api/template.pandoc \
#          --output=../api/${filename%.*}.html
# done
