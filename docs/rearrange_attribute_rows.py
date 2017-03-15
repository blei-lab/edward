"""Place attribute names in Sphinx blocks in a row above the attribute contents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

from bs4 import BeautifulSoup

path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  soup = BeautifulSoup(open(filename), 'html.parser')
  for tr in soup.find_all("tr", {'class': 'field'}):
    th = tr.find("th", {'class': 'field-name'})
    new_tr = soup.new_tag("tr", **{'class': 'field'})
    th = th.wrap(new_tr)
    tr.insert_before(th)

  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
