"""Strip paragraphs in lists in pandoc's html output."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

from bs4 import BeautifulSoup

paths = ("build/*.html", "build/api/*.html", "build/tutorials/*.html")
filenames = []

for path in paths:
  filenames.extend(glob.glob(path))

for filename in filenames:
  soup = BeautifulSoup(open(filename), 'html.parser')
  all_li = soup.find_all('li')
  if all_li:
    for list_item in all_li:
      if list_item.p is not None:
        list_item.p.unwrap()
    html = str(soup)
    html = html.replace('border="1"', '')
    with open(filename, 'wb') as file:
      file.write(html)
