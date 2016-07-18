from bs4 import BeautifulSoup
import glob
path = "*.html"
for filename in glob.glob(path):
  soup = BeautifulSoup(open(filename),'html.parser')
  all_li = soup.find_all('li')
  if all_li:
    for list_item in all_li:
      list_item.p.unwrap()
    html = str(soup)
    with open(filename, 'wb') as file:
        file.write(html)
