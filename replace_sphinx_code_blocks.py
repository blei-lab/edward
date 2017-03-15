"""Format Sphinx code blocks as Python; trim '>>>', '...', and last '\n'."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re

from bs4 import BeautifulSoup

path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  soup = BeautifulSoup(open(filename), 'html.parser')
  for snippet in soup.find_all("div", class_="highlight-default"):
    snippet.name = "pre"
    snippet["class"] = "python"
    snippet["language"] = "Python"
    raw_snippet = snippet.get_text()
    raw_snippet = re.sub('>>> ', '', raw_snippet)
    raw_snippet = re.sub('>>>', '', raw_snippet)
    raw_snippet = re.sub('\.\.\. ', '', raw_snippet)
    raw_snippet = re.sub('\.\.\.', '', raw_snippet)
    if raw_snippet[-2:] == '\n\n':
      raw_snippet = raw_snippet[:-1]

    tag_snippet = soup.new_tag("code")
    tag_snippet.string = raw_snippet
    snippet.find_all("div", class_="highlight")[0].replace_with(tag_snippet)

  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
