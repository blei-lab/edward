"""Remove 'Methods' <p> in Sphinx code blocks if there are no methods.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bs4
import glob

from bs4 import BeautifulSoup

path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  soup = BeautifulSoup(open(filename), 'html.parser')
  for dd in soup.find_all("dd"):
    try:
      # get last Tag element
      for last_content in reversed(dd.contents):
        if isinstance(last_content, bs4.element.Tag):
          break
    except:
      pass

    # remove if last tag is Methods
    if last_content.name == 'p' and \
       last_content.get('class') == ["rubric"] and \
       last_content.text == "Methods":
      last_content.extract()

  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
