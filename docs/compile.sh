#!/bin/bash
printf "Compiling Edward website.\n\n"

echo "Clearing all html files."
rm -f {./,api/,tutorials/}*.html
printf "Done.\n\n"

# Generate docstrings
echo "Begin docstring generation."
sphinx-build -b html -j 4 "tex/api/" "build/html/"
python autogen.py
printf "Done.\n\n"

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
  echo api/$filename
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
printf "Done.\n\n"

cd ..
echo "Begin postprocessing scripts."
echo "./insert_github_links.py"
python insert_github_links.py
echo "./rearrange_attribute_rows.py"
python rearrange_attribute_rows.py
echo "./remove_orphan_methods.py"
python remove_orphan_methods.py
echo "./replace_sphinx_code_blocks.py"
python replace_sphinx_code_blocks.py
echo "./strip_p_in_li.py"
python strip_p_in_li.py
printf "Done.\n\n"

# Clear intermediate docstring-generated files
rm -rf build
