#!/bin/bash
printf "Compiling Edward website.\n\n"
docdir=$(pwd)

echo "Clearing any previously built files."
rm -rf build/
printf "Done.\n\n"

echo "Begin docstring generation."
python parser/generate.py \
  --src_dir=$docdir/tex/ \
  --output_dir=/tmp/docs/
python autogen.py \
  --src_dir=/tmp/docs/
printf "Done.\n\n"

echo "Begin pandoc compilation."
mkdir -p build/api
mkdir -p build/tutorials
mkdir -p $docdir/build/api/ed/{criticisms/ppc_plots/,inferences/,models/}
cd /tmp/docs/api
# TODO recursively
for filename in {./,ed/,ed/criticisms/,ed/criticisms/ppc_plots/}*.md; do
  echo api/$filename
  pandoc ${filename%.*}.md \
         --from=markdown+link_attributes+native_spans \
         --to=html \
         --filter=$docdir/pandoc-code2raw.py \
         --mathjax \
         --no-highlight \
         --bibliography=$docdir/tex/bib.bib \
         --csl=$docdir/tex/apa.csl \
         --title-prefix="Edward" \
         --template=$docdir/tex/template.pandoc \
         --output=$docdir/build/api/${filename%.*}.html
done
cd ..
for filename in {./,api/,tutorials/}*.tex; do
  echo $filename
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter=$docdir/pandoc-code2raw.py \
         --mathjax \
         --no-highlight \
         --bibliography=$docdir/tex/bib.bib \
         --csl=$docdir/tex/apa.csl \
         --title-prefix="Edward" \
         --template=$docdir/tex/template.pandoc \
         --output=$docdir/build/${filename%.*}.html
done
printf "Done.\n\n"

cd $docdir
echo "Begin postprocessing scripts."
echo "./strip_p_in_li.py"
python strip_p_in_li.py
printf "Done.\n\n"

echo "Begin copying index files."
cp -r css/ build/
cp -r icons/ build/
cp -r images/ build/
cp CNAME build/
printf "Done.\n\n"

# Clear intermediate docstring-generated files
rm -rf /tmp/docs
