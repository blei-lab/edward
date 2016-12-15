from bs4 import BeautifulSoup
import glob

print "Running `strip_p_in_li.py`"

paths = ("*.html", "api/*.html", "tutorials/*.html")
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
