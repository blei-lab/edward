#!/bin/bash
printf "Compiling Edward website.\n\n"
docdir=$(pwd)
outdir=$docdir/build
tmpdir=/tmp/docs
tmpdir2=/tmp/docs2

echo "Clearing any previously built files."
rm -rf "$outdir/"
printf "Done.\n\n"

echo "Begin docstring generation."
python generate_api_navbar_and_symbols.py \
  --src_dir="$docdir/tex/" \
  --out_dir="$tmpdir2/"
python parser/generate.py \
  --src_dir="$tmpdir2/" \
  --output_dir="$tmpdir/"
python generate_api_toc.py \
    --src_dir="$docdir/tex/template-api.pandoc" \
    --yaml_dir="$tmpdir/api/_toc.yaml" \
    --out_dir="$tmpdir/template.pandoc"
printf "Done.\n\n"

echo "Begin pandoc compilation."
cd "$tmpdir"
for filename in $(find api -name '*.md'); do
  echo "$filename"
  mkdir -p "$outdir/$(dirname $filename)"
  if [[ "$filename" == api/observations* ]]; then
    # assume observations/ lives in same parent directory as edward/
    bib="$docdir/../../observations/bib.bib"
  else
    bib="$docdir/tex/bib.bib"
  fi
  pandoc "$filename" \
         --from=markdown+link_attributes+native_spans \
         --to=html \
         --filter="$docdir/pandoc-code2raw.py" \
         --mathjax \
         --no-highlight \
         --bibliography="$bib" \
         --csl="$docdir/tex/apa.csl" \
         --title-prefix="Edward" \
         --template="$tmpdir/template.pandoc" \
         --output="$outdir/${filename%.*}.html"
done
for filename in $(find api -name '*.tex'); do
  echo "$filename"
  mkdir -p "$outdir/$(dirname $filename)"
  pandoc "$filename" \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter="$docdir/pandoc-code2raw.py" \
         --mathjax \
         --no-highlight \
         --bibliography="$docdir/tex/bib.bib" \
         --csl="$docdir/tex/apa.csl" \
         --title-prefix="Edward" \
         --template="$tmpdir/template.pandoc" \
         --output="$outdir/${filename%.*}.html"
done
for filename in {./,tutorials/}*.tex; do
  echo "$filename"
  mkdir -p "$outdir/$(dirname $filename)"
  pandoc "$filename" \
         --from=latex+link_attributes+native_spans \
         --to=html \
         --filter="$docdir/pandoc-code2raw.py" \
         --mathjax \
         --no-highlight \
         --bibliography="$docdir/tex/bib.bib" \
         --csl="$docdir/tex/apa.csl" \
         --title-prefix="Edward" \
         --template="$docdir/tex/template.pandoc" \
         --output="$outdir/${filename%.*}.html"
done
printf "Done.\n\n"

echo "Begin postprocessing scripts."
cd "$docdir"
echo "./strip_p_in_li.py"
python "strip_p_in_li.py"
printf "Done.\n\n"

echo "Begin copying index files."
cp -r "css/" "$outdir/"
cp -r "icons/" "$outdir/"
cp -r "images/" "$outdir/"
cp "CNAME" "$outdir/"
printf "Done.\n\n"

# Clear intermediate docstring-generated files
rm -rf "$tmpdir"
rm -rf "$tmpdir2"
