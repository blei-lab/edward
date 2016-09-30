#!/bin/bash
echo "Compiling website."

echo "Clearing all html files."
rm -f *.html
rm -rf api
rm -rf tutorials

# Generate docstrings
echo "Begin docstring generation."
mkdir -p build
python autogen.py

# Compile all the tex files into html
echo "Begin pandoc compilation."
mkdir -p api
mkdir -p tutorials
cd tex
for filename in {./,tutorials/}*.tex; do
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
cd ../build
for filename in *.tex; do
  echo $filename
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter="../pandoc-code2raw.py" \
         --mathjax \
         --no-highlight \
         --bibliography=../tex/bib.bib \
         --csl=../tex/apa.csl \
         --title-prefix="Edward" \
         --template=../tex/template.pandoc \
         --output=../api/${filename%.*}.html
done

# Strip paragraphs in lists in pandoc's html output
cd ..
python strip_p_in_li.py

# Clear intermediate docstring-generated files
rm -rf build
