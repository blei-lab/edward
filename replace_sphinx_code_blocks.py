from bs4 import BeautifulSoup
import glob

print "Running `replace_sphinx_code_blocks.py`"

path = ("api/*.html")
filenames = glob.glob(path)

for filename in filenames:
  print filename
  soup = BeautifulSoup(open(filename), 'html.parser')
  for snippet in soup.find_all("div", class_="highlight-default"):
    snippet.name = "pre"
    snippet["class"] = "python"
    snippet["language"] = "Python"
    raw_snippet = snippet.get_text()
    tag_snippet = soup.new_tag("code")
    tag_snippet.string = raw_snippet
    snippet.find_all("div", class_="highlight")[0].replace_with(tag_snippet)
  html = str(soup)
  with open(filename, 'wb') as file:
    file.write(html)
