"""Replace paragraph symbol with link to github with line numbering."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward
import inspect
import glob
import string

from bs4 import BeautifulSoup


def find_filename_and_line(method_name):
  components = string.split(method_name, '.')
  method_obj = edward
  for i in range(1, len(components)):
    method_obj = getattr(method_obj, components[i])

  rel_path = inspect.getsourcefile(method_obj).split("edward/", 1)[1]
  line_no = inspect.getsourcelines(method_obj)[-1]
  return rel_path, line_no


path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  soup = BeautifulSoup(open(filename), 'html.parser')
  github = "https://github.com/blei-lab/edward/blob/master/"
  for a in soup.find_all("a", "headerlink"):
    try:
      rel_path, line_no = find_filename_and_line(a['href'][1:])
      link = github + rel_path + '#L' + str(line_no)
      a['href'] = link
      a['class'] = 'u-pull-right'
      a['title'] = "Link to definition on GitHub."
      a.string.replace_with("[source]")
    except:  # e.g., if method_obj is a property
      a['href'] = ""
      a.string.replace_with("")

  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
