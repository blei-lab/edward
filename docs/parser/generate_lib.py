# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate docs for the TensorFlow Python API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import six

from tensorflow.python.util import tf_inspect
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse
from tensorflow.tools.docs import doc_generator_visitor
from tensorflow.tools.docs import parser
from tensorflow.tools.docs import pretty_docs
from tensorflow.tools.docs import py_guide_parser


def _is_free_function(py_object, full_name, index):
  """Check if input is a free function (and not a class- or static method)."""
  if not tf_inspect.isfunction(py_object):
    return False

  # Static methods are functions to tf_inspect (in 2.7), so check if the parent
  # is a class. If there is no parent, it's not a function.
  if '.' not in full_name:
    return False

  parent_name = full_name.rsplit('.', 1)[0]
  if tf_inspect.isclass(index[parent_name]):
    return False

  return True


def write_docs(output_dir, parser_config, yaml_toc):
  """Write previously extracted docs to disk.

  Write a docs page for each symbol included in the indices of parser_config to
  a tree of docs at `output_dir`.

  Symbols with multiple aliases will have only one page written about
  them, which is referenced for all aliases.

  Args:
    output_dir: Directory to write documentation markdown files to. Will be
      created if it doesn't exist.
    parser_config: A `parser.ParserConfig` object, containing all the necessary
      indices.
    yaml_toc: Set to `True` to generate a "_toc.yaml" file.

  Raises:
    ValueError: if `output_dir` is not an absolute path
  """
  # Make output_dir.
  if not os.path.isabs(output_dir):
    raise ValueError(
        "'output_dir' must be an absolute path.\n"
        "    output_dir='%s'" % output_dir)

  try:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  except OSError as e:
    print('Creating output dir "%s" failed: %s' % (output_dir, e))
    raise

  # These dictionaries are used for table-of-contents generation below
  # They will contain, after the for-loop below::
  #  - module name(string):classes and functions the module contains(list)
  module_children = {}
  #  - symbol name(string):pathname (string)
  symbol_to_file = {}

  # Parse and write Markdown pages, resolving cross-links (@{symbol}).
  for full_name, py_object in six.iteritems(parser_config.index):

    if full_name in parser_config.duplicate_of:
      continue

    # Methods and some routines are documented only as part of their class.
    if not (tf_inspect.ismodule(py_object) or tf_inspect.isclass(py_object) or
            _is_free_function(py_object, full_name, parser_config.index)):
      continue

    sitepath = os.path.join('api_docs/python',
                            parser.documentation_path(full_name)[:-3])

    # For TOC, we need to store a mapping from full_name to the file
    # we're generating
    symbol_to_file[full_name] = sitepath

    # For a module, remember the module for the table-of-contents
    if tf_inspect.ismodule(py_object):
      if full_name in parser_config.tree:
        module_children.setdefault(full_name, [])

    # For something else that's documented,
    # figure out what module it lives in
    else:
      subname = str(full_name)
      while True:
        subname = subname[:subname.rindex('.')]
        if tf_inspect.ismodule(parser_config.index[subname]):
          module_children.setdefault(subname, []).append(full_name)
          break

    print('Writing docs for %s (%r).' % (full_name, py_object))

    # Generate docs for `py_object`, resolving references.
    page_info = parser.docs_for_object(full_name, py_object, parser_config)

    path = os.path.join(output_dir, parser.documentation_path(full_name))
    directory = os.path.dirname(path)
    try:
      if not os.path.exists(directory):
        os.makedirs(directory)
      with open(path, 'w') as f:
        f.write(pretty_docs.build_md_page(page_info))
    except OSError as e:
      print('Cannot write documentation for %s to %s: %s' % (full_name,
                                                             directory, e))
      raise

  if yaml_toc:
    # Generate table of contents

    # Put modules in alphabetical order, case-insensitive
    modules = sorted(module_children.keys(), key=lambda a: a.upper())

    leftnav_path = os.path.join(output_dir, '_toc.yaml')
    with open(leftnav_path, 'w') as f:

      # Generate header
      f.write('# Automatically generated file; please do not edit\ntoc:\n')
      for module in modules:
        f.write('  - title: ' + module + '\n'
                '    section:\n' + '    - title: Overview\n' +
                '      path: /TARGET_DOC_ROOT/VERSION/' + symbol_to_file[module]
                + '\n')

        symbols_in_module = module_children.get(module, [])
        # Sort case-insensitive, if equal sort case sensitive (upper first)
        symbols_in_module.sort(key=lambda a: (a.upper(), a))

        for full_name in symbols_in_module:
          f.write('    - title: ' + full_name[len(module) + 1:] + '\n'
                  '      path: /TARGET_DOC_ROOT/VERSION/' +
                  symbol_to_file[full_name] + '\n')

  # Write a global index containing all full names with links.
  with open(os.path.join(output_dir, 'index.md'), 'w') as f:
    f.write(
        parser.generate_global_index('TensorFlow', parser_config.index,
                                     parser_config.reference_resolver))


def add_dict_to_dict(add_from, add_to):
  for key in add_from:
    if key in add_to:
      add_to[key].extend(add_from[key])
    else:
      add_to[key] = add_from[key]


# Exclude some libaries in contrib from the documentation altogether.
def _get_default_private_map():
  return {}


# Exclude members of some libaries.
def _get_default_do_not_descend_map():
  # TODO(wicke): Shrink this list once the modules get sealed.
  return {
      'tf': ['cli', 'lib', 'wrappers'],
      'tf.contrib': [
          'compiler',
          'factorization',
          'grid_rnn',
          'labeled_tensor',
          'ndlstm',
          'quantization',
          'session_bundle',
          'slim',
          'solvers',
          'specs',
          'tensor_forest',
          'tensorboard',
          'testing',
          'tfprof',
      ],
      'tf.contrib.bayesflow': [
          'special_math', 'stochastic_gradient_estimators',
          'stochastic_variables'
      ],
      'tf.contrib.ffmpeg': ['ffmpeg_ops'],
      'tf.contrib.graph_editor': [
          'edit', 'match', 'reroute', 'subgraph', 'transform', 'select', 'util'
      ],
      'tf.contrib.keras': ['api', 'python'],
      'tf.contrib.layers': ['feature_column', 'summaries'],
      'tf.contrib.learn': [
          'datasets',
          'head',
          'graph_actions',
          'io',
          'models',
          'monitors',
          'ops',
          'preprocessing',
          'utils',
      ],
      'tf.contrib.util': ['loader'],
  }


def extract(py_modules, private_map, do_not_descend_map):
  """Extract docs from tf namespace and write them to disk."""
  # Traverse the first module.
  visitor = doc_generator_visitor.DocGeneratorVisitor(py_modules[0][0])
  api_visitor = public_api.PublicAPIVisitor(visitor)
  api_visitor.set_root_name(py_modules[0][0])
  add_dict_to_dict(private_map, api_visitor.private_map)
  add_dict_to_dict(do_not_descend_map, api_visitor.do_not_descend_map)

  traverse.traverse(py_modules[0][1], api_visitor)

  # Traverse all py_modules after the first:
  for module_name, module in py_modules[1:]:
    visitor.set_root_name(module_name)
    api_visitor.set_root_name(module_name)
    traverse.traverse(module, api_visitor)

  return visitor


class _GetMarkdownTitle(py_guide_parser.PyGuideParser):
  """Extract the title from a .md file."""

  def __init__(self):
    self.title = None
    py_guide_parser.PyGuideParser.__init__(self)

  def process_title(self, _, title):
    if self.title is None:  # only use the first title
      self.title = title


class _DocInfo(object):
  """A simple struct for holding a doc's url and title."""

  def __init__(self, url, title):
    self.url = url
    self.title = title


def build_doc_index(src_dir):
  """Build an index from a keyword designating a doc to _DocInfo objects."""
  doc_index = {}
  if not os.path.isabs(src_dir):
    raise ValueError("'src_dir' must be an absolute path.\n"
                     "    src_dir='%s'" % src_dir)

  if not os.path.exists(src_dir):
    raise ValueError("'src_dir' path must exist.\n"
                     "    src_dir='%s'" % src_dir)

  for dirpath, _, filenames in os.walk(src_dir):
    suffix = os.path.relpath(path=dirpath, start=src_dir)
    for base_name in filenames:
      if not base_name.endswith('.md'):
        continue
      title_parser = _GetMarkdownTitle()
      title_parser.process(os.path.join(dirpath, base_name))
      key_parts = os.path.join(suffix, base_name[:-3]).split('/')
      if key_parts[-1] == 'index':
        key_parts = key_parts[:-1]
      doc_info = _DocInfo(os.path.join(suffix, base_name), title_parser.title)
      doc_index[key_parts[-1]] = doc_info
      if len(key_parts) > 1:
        doc_index['/'.join(key_parts[-2:])] = doc_info

  return doc_index


class _GuideRef(object):

  def __init__(self, base_name, title, section_title, section_tag):
    self.url = 'api_guides/python/' + (('%s#%s' % (base_name, section_tag))
                                       if section_tag else base_name)
    self.link_text = (('%s > %s' % (title, section_title))
                      if section_title else title)

  def make_md_link(self, url_prefix):
    return '[%s](%s%s)' % (self.link_text, url_prefix, self.url)


class _GenerateGuideIndex(py_guide_parser.PyGuideParser):
  """Turn guide files into an index from symbol name to a list of _GuideRefs."""

  def __init__(self):
    self.index = {}
    py_guide_parser.PyGuideParser.__init__(self)

  def process(self, full_path, base_name):
    """Index a file, reading from `full_path`, with `base_name` as the link."""
    self.full_path = full_path
    self.base_name = base_name
    self.title = None
    self.section_title = None
    self.section_tag = None
    py_guide_parser.PyGuideParser.process(self, full_path)

  def process_title(self, _, title):
    if self.title is None:  # only use the first title
      self.title = title

  def process_section(self, _, section_title, tag):
    self.section_title = section_title
    self.section_tag = tag

  def process_line(self, _, line):
    """Index @{symbol} references as in the current file & section."""
    for match in parser.SYMBOL_REFERENCE_RE.finditer(line):
      val = self.index.get(match.group(1), [])
      val.append(
          _GuideRef(self.base_name, self.title, self.section_title,
                    self.section_tag))
      self.index[match.group(1)] = val


def _build_guide_index(guide_src_dir):
  """Return dict: symbol name -> _GuideRef from the files in `guide_src_dir`."""
  index_generator = _GenerateGuideIndex()
  if os.path.exists(guide_src_dir):
    for full_path, base_name in py_guide_parser.md_files_in_dir(guide_src_dir):
      index_generator.process(full_path, base_name)
  return index_generator.index


class _UpdateTags(py_guide_parser.PyGuideParser):
  """Rewrites a Python guide so that each section has an explicit tag."""

  def process_section(self, line_number, section_title, tag):
    self.replace_line(line_number, '<h2 id="%s">%s</h2>' % (tag, section_title))


EXCLUDED = set(['__init__.py', 'OWNERS', 'README.txt'])


def _other_docs(src_dir, output_dir, reference_resolver):
  """Convert all the files in `src_dir` and write results to `output_dir`."""
  header = '<!-- DO NOT EDIT! Automatically generated file. -->\n'

  # Iterate through all the source files and process them.
  tag_updater = _UpdateTags()
  for dirpath, _, filenames in os.walk(src_dir):
    # How to get from `dirpath` to api_docs/python/
    relative_path_to_root = os.path.relpath(
        path=os.path.join(src_dir, 'api_docs/python'), start=dirpath)

    # Make the directory under output_dir.
    new_dir = os.path.join(output_dir,
                           os.path.relpath(path=dirpath, start=src_dir))
    try:
      if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    except OSError as e:
      print('Creating output dir "%s" failed: %s' % (new_dir, e))
      raise

    for base_name in filenames:
      if base_name in EXCLUDED:
        print('Skipping excluded file %s...' % base_name)
        continue
      full_in_path = os.path.join(dirpath, base_name)
      suffix = os.path.relpath(path=full_in_path, start=src_dir)
      full_out_path = os.path.join(output_dir, suffix)
      if not base_name.endswith('.md'):
        print('Copying non-md file %s...' % suffix)
        open(full_out_path, 'w').write(open(full_in_path).read())
        continue
      if dirpath.endswith('/api_guides/python'):
        print('Processing Python guide %s...' % base_name)
        md_string = tag_updater.process(full_in_path)
      else:
        print('Processing doc %s...' % suffix)
        md_string = open(full_in_path).read()

      output = reference_resolver.replace_references(md_string,
                                                     relative_path_to_root)
      with open(full_out_path, 'w') as f:
        f.write(header + output)

  print('Done.')


class DocGenerator(object):
  """Main entry point for generating docs."""

  def __init__(self):
    self.argument_parser = argparse.ArgumentParser()
    self._py_modules = None
    self._private_map = _get_default_private_map()
    self._do_not_descend_map = _get_default_do_not_descend_map()
    self.yaml_toc = True

  def add_output_dir_argument(self):
    self.argument_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        required=True,
        help='Directory to write docs to.')

  def add_src_dir_argument(self):
    self.argument_parser.add_argument(
        '--src_dir',
        type=str,
        default=None,
        required=True,
        help='Directory with the source docs.')

  def add_base_dir_argument(self, default_base_dir):
    self.argument_parser.add_argument(
        '--base_dir',
        type=str,
        default=default_base_dir,
        help='Base directory to to strip from file names referenced in docs.')

  def parse_known_args(self):
    flags, _ = self.argument_parser.parse_known_args()
    return flags

  def add_to_private_map(self, d):
    add_dict_to_dict(d, self._private_map)

  def add_to_do_not_descend_map(self, d):
    add_dict_to_dict(d, self._do_not_descend_map)

  def set_private_map(self, d):
    self._private_map = d

  def set_do_not_descend_map(self, d):
    self._do_not_descend_map = d

  def set_py_modules(self, py_modules):
    self._py_modules = py_modules

  def py_module_names(self):
    if self._py_modules is None:
      raise RuntimeError(
          'Must call set_py_modules() before running py_module_names().')
    return [name for (name, _) in self._py_modules]

  def make_reference_resolver(self, visitor, doc_index):
    return parser.ReferenceResolver.from_visitor(
        visitor, doc_index, py_module_names=self.py_module_names())

  def make_parser_config(self, visitor, reference_resolver, guide_index,
                         base_dir):
    return parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates=visitor.duplicates,
        duplicate_of=visitor.duplicate_of,
        tree=visitor.tree,
        index=visitor.index,
        reverse_index=visitor.reverse_index,
        guide_index=guide_index,
        base_dir=base_dir)

  def run_extraction(self):
    return extract(
        self._py_modules, self._private_map, self._do_not_descend_map)

  def build(self, flags):
    """Actually build the docs."""
    doc_index = build_doc_index(flags.src_dir)
    visitor = self.run_extraction()
    reference_resolver = self.make_reference_resolver(visitor, doc_index)

    guide_index = _build_guide_index(
        os.path.join(flags.src_dir, 'api_guides/python'))

    parser_config = self.make_parser_config(visitor, reference_resolver,
                                            guide_index, flags.base_dir)
    output_dir = os.path.join(flags.output_dir, 'api_docs/python')

    write_docs(output_dir, parser_config, yaml_toc=self.yaml_toc)
    _other_docs(flags.src_dir, flags.output_dir, reference_resolver)

    if parser.all_errors:
      print('Errors during processing:\n  ' + '\n  '.join(parser.all_errors))
      return 1
    return 0
