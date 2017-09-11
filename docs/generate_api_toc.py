"""Take generated TOC YAML file and format it into template.pandoc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--yaml_dir', type=str)
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()

src_dir = os.path.expanduser(args.src_dir)
yaml_dir = os.path.expanduser(args.yaml_dir)
out_dir = os.path.expanduser(args.out_dir)

with open(yaml_dir) as f:
  data_map = yaml.safe_load(f)

toc = ''
for entry in data_map['toc']:
  title = entry['title']
  if title == 'ed':
    continue

  section = entry['section']
  assert section[0]['title'] == 'Overview'
  path = section[0]['path']
  toc += '<a class="button u-full-width" href="{}">{}</a>'.format(path, title)
  toc += '\n'

document = open(src_dir).read()
document = document.replace('{{toc}}', toc)
open(out_dir, 'w').write(document)
