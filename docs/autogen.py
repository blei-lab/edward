"""
Autogenerate navbar and docstrings.

All pages in tex/api/ must be an element in PAGES. Otherwise the
page will have no navbar (or docstrings).

The order of the navbar is given by the order of PAGES.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import re

# Note we don't strictly need the 'parent_pages' field. you can
# technically infer them based on the other pages' 'child_pages'. It
# is denoted only for convenience.
PAGES = [
    {
        'page': 'index.tex',
        'title': 'Home',
        'parent_pages': [],
        'child_pages': [],
    },
    {
        'page': 'data.tex',
        'title': 'Data',
        'parent_pages': [],
        'child_pages': [],
    },
    {
        'page': 'model.tex',
        'title': 'Model',
        'parent_pages': [],
        'child_pages': [
            'model-compositionality.tex',
            'model-wrappers.tex',
            'model-development.tex',
        ],
    },
    {
        'page': 'model-compositionality.tex',
        'title': 'Compositionality',
        'parent_pages': [
            'model.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'model-development.tex',
        'title': 'Development',
        'parent_pages': [
            'model.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'model-wrappers.tex',
        'title': 'Wrappers',
        'parent_pages': [
            'model.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'inference.tex',
        'title': 'Inference',
        'parent_pages': [],
        'child_pages': [
            'inference-classes.tex',
            'inference-compositionality.tex',
            'inference-data-subsampling.tex',
            'inference-development.tex',
        ],
    },
    {
        'page': 'inference-classes.tex',
        'title': 'Classes',
        'parent_pages': [
            'inference.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'inference-compositionality.tex',
        'title': 'Compositionality',
        'parent_pages': [
            'inference.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'inference-data-subsampling.tex',
        'title': 'Data Subsampling',
        'parent_pages': [
            'inference.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'inference-development.tex',
        'title': 'Development',
        'parent_pages': [
            'inference.tex'
        ],
        'child_pages': [],
    },
    {
        'page': 'criticism.tex',
        'title': 'Criticism',
        'parent_pages': [],
        'child_pages': [],
    },
    # {
    #     'page': 'util.tex',
    #     'title': 'Utilities',
    #     'parent_pages': [],
    #     'child_pages': [],
    # },
]


def generate_navbar(page_data):
  def generate_top_navbar():
    # Create top of navbar. (Note this can be cached and not run within a loop.)
    top_navbar = """\\begin{abstract}
\subsection{API and Documentation}
\\begin{lstlisting}[raw=html]
<div class="row" style="padding-bottom: 5%">
<div class="row" style="padding-bottom: 1%">"""
    for page_data in PAGES:
      title = page_data['title']
      page_name = page_data['page']
      parent_pages = page_data['parent_pages']
      if len(parent_pages) == 0 and page_name != 'index.tex':
        top_navbar += '\n'
        top_navbar += '<a class="button3" href="/api/'
        top_navbar += page_name.replace('.tex', '')
        top_navbar += '">'
        top_navbar += title
        top_navbar += '</a>'

    top_navbar += '\n'
    top_navbar += '</div>'
    return top_navbar

  page_name = page_data['page']
  title = page_data['title']
  parent_pages = page_data['parent_pages']
  child_pages = page_data['child_pages']

  navbar = generate_top_navbar()
  # Create bottom of navbar if there are child pages for that section.
  if len(child_pages) > 0 or len(parent_pages) > 0:
    if len(parent_pages) > 0:
      parent = parent_pages[0]
      parent_page = [page_data for page_data in PAGES
                     if page_data['page'] == parent][0]
      pgs = parent_page['child_pages']
    else:
      pgs = child_pages

    navbar += '\n'
    navbar += '<div class="row">'
    for child_page in pgs:
      navbar += '\n'
      navbar += '<a class="button4" href="/api/'
      navbar += child_page.replace('.tex', '')
      navbar += '">'
      navbar += [page_data for page_data in PAGES
                 if page_data['page'] == child_page][0]['title']
      navbar += '</a>'

    navbar += '\n'
    navbar += '</div>'

  navbar += '\n'
  navbar += """</div>
\end{lstlisting}
\end{abstract}"""

  # Set primary button in navbar. If a child page, set primary buttons
  # for both top and bottom of navbar.
  search_term = '" href="/api/' + page_name.replace('.tex', '') + '">'
  navbar = navbar.replace(search_term, ' button-primary' + search_term)
  if len(parent_pages) > 0:
    parent = parent_pages[0]
    search_term = '" href="/api/' + parent.replace('.tex', '') + '">'
    navbar = navbar.replace(search_term, ' button-primary' + search_term)

  return navbar


def generate_docstrings(page_data):
  page_name = page_data['page'][:-4]
  path = os.path.join('build/html/', page_name + '.html')
  document = open(path).read()
  docstring = """\\begin{lstlisting}[raw=html]\n"""
  sphinx_html = re.search(r'(?<=<p>{{sphinx</p>)(.*?)(?=<p>}}</p>)',
                          document, flags=re.DOTALL)
  if sphinx_html:
    docstring += sphinx_html.group(0)
  docstring += """\\end{lstlisting}"""
  return docstring


print("Populating build/ directory with files from tex/api/.")
for subdir, dirs, fnames in os.walk('tex/api'):
  for fname in fnames:
    new_subdir = subdir.replace('tex/api', 'build')
    if not os.path.exists(new_subdir):
      os.makedirs(new_subdir)

    if fname[-4:] == '.tex':
      fpath = os.path.join(subdir, fname)
      new_fpath = fpath.replace('tex/api', 'build')
      shutil.copy(fpath, new_fpath)

print("Starting autogeneration.")
for page_data in PAGES:
  page_name = page_data['page']
  path = os.path.join('build', page_name)

  print("...generating navigation bar:", path)
  navbar = generate_navbar(page_data)

  print("...generating docstrings:", path)
  docstring = generate_docstrings(page_data)

  # Either insert content into existing page, or create page otherwise.
  if os.path.exists(path):
    document = open(path).read()
    assert '{{navbar}}' in document, \
           ("File found for " + path + " but missing {{navbar}} tag.")
    print("...inserting autogenerated content into file:", path)
    document = document.replace('{{navbar}}', navbar)
    document = re.sub(r'(?<={{sphinx)(.*?)(?=}})', "",
                      document, flags=re.DOTALL)
    document = document.replace('{{sphinx}}', docstring)
  else:
    print("...creating new file with autogenerated content:", path)
    document = '\title{'
    document += title
    document += '}'
    document += '\n' + '\n'
    document += navbar
    document += '\n' + '\n'
    document += '\subsubsection{'
    document += title
    document += '}'
    document += '\n' + '\n'
    document += docstring

  subdir = os.path.dirname(path)
  if not os.path.exists(subdir):
    os.makedirs(subdir)

  print("...writing file:", path)
  open(path, 'w').write(document)
