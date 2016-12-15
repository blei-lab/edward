from bs4 import BeautifulSoup
import glob
import edward
import inspect
import string


def find_filename_and_line(method_name):
  components = string.split(method_name, '.')
  if len(components) == 3:
    method_obj = getattr(getattr(edward, components[1]), components[2])
  if len(components) == 4:
    method_obj = getattr(getattr(getattr(edward, components[1]),
                                 components[2]),
                         components[3])
  rel_path = inspect.getsourcefile(method_obj).split("edward/", 1)[1]
  line_no = inspect.getsourcelines(method_obj)[-1]
  return rel_path, line_no

print "Running `insert_github_links.py`"

path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  print filename
  soup = BeautifulSoup(open(filename), 'html.parser')
  github = "https://github.com/blei-lab/edward/blob/master/"
  for a in soup.find_all("a", "headerlink"):
    rel_path, line_no = find_filename_and_line(a['href'][1:])
    link = github + rel_path + '#L' + str(line_no)
    a['href'] = link
    a['class'] = 'u-pull-right'
    a['title'] = "Link to definition on GitHub."
    a.string.replace_with("[source]")
  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
