from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def maybe_download_and_extract(directory, url, extract=True):
  """Download file from url unless it already exists in specified directory.
  Extract the file if `extract` is True.

  The file at `url` is downloaded to the directory `directory`
  with its original filename. For example, with url
  `http://example.org/example.txt` and directory `~/data`, the
  downloaded file is located at `~/data/example.txt`.

  Args:
    directory: str.
      Path to directory containing the file or where file will be downloaded.
    url: str.
      URL to download from if file doesn't exist.
    extract: bool, optional.
      If True, tries to extract the file if it has format 'gz',
      'tar' (including 'tar.gz' and 'tar.bz'), or 'zip'.

  Returns:
    str. Path to downloaded or already existing file.
  """
  import gzip
  import os
  import sys
  import tarfile
  import zipfile
  from six.moves import urllib
  directory = os.path.expanduser(directory)
  if not os.path.exists(directory):
    os.makedirs(directory)

  filename = url.split('/')[-1]
  filepath = os.path.join(directory, filename)
  if os.path.exists(filepath):
    return filepath

  def _progress(count, block_size, total_size):
    sys.stdout.write(
        '\r>> Downloading %s %.1f%%' %
        (filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  try:
    urllib.request.urlretrieve(url, filepath, _progress)
  except:
    if os.path.exists(filepath):
      os.remove(filepath)
    raise
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  if extract:
    if tarfile.is_tarfile(filepath):
      with tarfile.open(filepath) as f:
        f.extractall(directory)
    elif filename.endswith('.gz'):
      with gzip.open(filepath, 'rb') as f:
        s = f.read()
      extracted_filepath = os.path.splitext(filepath)[0]
      with open(extracted_filepath, 'w') as f:
        f.write(s)
    elif zipfile.is_zipfile(filepath):
      with zipfile.ZipFile(filepath) as f:
        f.extractall(directory)

  return filepath
