# Parser

These are tools to parse docstrings into Markdown.

We build on the parser from TensorFlow
(commit `3cb3cc6d426112f432db62db6f493ea00ce31e0f`):

+ https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docs
+ https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/common

Use a diff tool to analyze the line-by-line differences.

## Usage

The main command is
```python
python parser/generate.py \
  --src_dir=/absolute/path/to/docs/tex/ \
  --output_dir=/tmp/docs/
```
It builds the API pages into `/tmp/docs/api/`. And it will copy all
non-MD and non-tex files from `tex/` into `/tmp/docs/` and
appropriately make code references.
