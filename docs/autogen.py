"""Autogenerate navbar.

All pages in src_dir/api/ must be an element in PAGES. Otherwise the
page will have no navbar.

The order of the navbar is given by the order of PAGES.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import re

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
args = parser.parse_args()

# Note we don't strictly need the 'parent_pages' field. We can
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
    {
        'page': 'reference.tex',
        'title': 'Reference',
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
  """Return a string. It is the navigation bar for ``page_data``."""
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


def generate_tensorflow_distributions():
  import edward.models
  from tensorflow.contrib import distributions

  built_in_models = ['DirichletProcess', 'distributions_DirichletProcess',
                     'Empirical', 'distributions_Empirical',
                     'ParamMixture', 'distributions_ParamMixture',
                     'PointMass', 'distributions_PointMass']
  models = [getattr(edward.models, name) for name in dir(edward.models)]
  models = [model for model in models
            if (isinstance(model, type) and
                issubclass(model, distributions.Distribution) and
                model.__name__ not in built_in_models
                )
            ]
  models = sorted(models, key=lambda cls: cls.__name__)

  stub = 'https://www.tensorflow.org/api_docs/python/tf/contrib/distributions'
  fragment = ('\\begin{{lstlisting}}[raw=html]'
              '<a class="reference" '
              'href="{stub}/{name}" title="edward.models.{name}">'
              '<code class="xref py py-class docutils literal">'
              '<span class="pre">ed.models.{name}</span>'
              '</code>'
              '</a>'
              '\\end{{lstlisting}}')

  links = [fragment.format(stub=stub, name=cls.__name__)
           for cls in models]

  # note the start and end li tag are provided from outside
  return '\n\item'.join(links)


def get_tensorflow_version():
  import tensorflow
  return str(getattr(tensorflow, '__version__', '<unknown verison>'))


print("Starting autogeneration.")
for page_data in PAGES:
  page_name = page_data['page']
  path = os.path.join(args.src_dir, 'api', page_name)
  print(path)

  # Generate navigation bar.
  navbar = generate_navbar(page_data)

  # Insert autogenerated content into page.
  document = open(path).read()
  assert '{{navbar}}' in document, \
         ("File found for " + path + " but missing {{navbar}} tag.")
  document = document.replace('{{navbar}}', navbar)

  if '{{tensorflow_distributions}}' in document:
    document = document.replace('{{tensorflow_distributions}}',
                                generate_tensorflow_distributions())

  if '{{tensorflow_version}}' in document:
    document = document.replace('{{tensorflow_version}}',
                                get_tensorflow_version())

  subdir = os.path.dirname(path)
  if not os.path.exists(subdir):
    os.makedirs(subdir)

  # Write to file.
  open(path, 'w').write(document)
