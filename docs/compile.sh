#!/bin/bash
printf "Compiling Edward website.\n\n"
docdir=$(pwd)
tmpdir=/tmp/docs

echo "Clearing any previously built files."
rm -rf build/
printf "Done.\n\n"

echo "Begin docstring generation."
python parser/generate.py \
  --src_dir=$docdir/tex/ \
  --output_dir=$tmpdir/
python generate_api_navbar.py \
  --src_dir=$tmpdir/
python generate_api_toc.py \
    --src_dir=$docdir/tex/template-api.pandoc \
    --yaml_dir=$tmpdir/api/_toc.yaml \
    --out_dir=$tmpdir/template.pandoc
printf "Done.\n\n"

echo "Begin pandoc compilation."
# TODO recursively
mkdir -p build/api/ed/{criticisms/,inferences/,models/,util/}
mkdir -p build/tutorials
cd $tmpdir/api
for filename in {./,ed/,ed/criticisms/,ed/inferences/,ed/models/,ed/util/}*.md; do
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
         --template=$tmpdir/template.pandoc \
         --output=$docdir/build/api/${filename%.*}.html
done
for filename in *.tex; do
  echo api/$filename
  pandoc ${filename%.*}.tex \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter=$docdir/pandoc-code2raw.py \
         --mathjax \
         --no-highlight \
         --bibliography=$docdir/tex/bib.bib \
         --csl=$docdir/tex/apa.csl \
         --title-prefix="Edward" \
         --template=$tmpdir/template.pandoc \
         --output=$docdir/build/api/${filename%.*}.html
done
cd ..
for filename in {./,tutorials/}*.tex; do
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
