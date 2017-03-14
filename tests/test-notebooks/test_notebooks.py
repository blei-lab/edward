from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import nbformat
import os
import sys
import time
import traceback
import unittest

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


class test_notebooks(unittest.TestCase):

  def _exec_notebook(self, ep, notebook_filename, nbpath):
    with open(notebook_filename) as f:
      nb = nbformat.read(f, as_version=nbformat.current_nbformat)
      try:
        out = ep.preprocess(nb, {'metadata': {'path': nbpath}})
      except CellExecutionError:
        print('-' * 60)
        traceback.print_exc(file=sys.stdout)
        print('-' * 60)
        self.assertTrue(False,
                        'Error executing the notebook %s. See above for error.'
                        % notebook_filename)

  def test_all_notebooks(self):
    """ Test all notebooks except blacklist. """
    blacklist = ['gan.ipynb']
    pythonkernel = 'python' + str(sys.version_info[0])
    nbpath = '../../docs/notebooks/'
    # see http://nbconvert.readthedocs.io/en/stable/execute_api.html
    ep = ExecutePreprocessor(timeout=120,
                             kernel_name=pythonkernel,
                             interrupt_on_timeout=True)
    lfiles = glob.glob(nbpath + "*.ipynb")
    for notebook_filename in lfiles:
      if(os.path.basename(notebook_filename) not in blacklist):
        t = time.time()
        self._exec_notebook(ep, notebook_filename, nbpath)
        print(notebook_filename, 'took %g seconds.' % (time.time() - t))

if __name__ == '__main__':
  unittest.main()
