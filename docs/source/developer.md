# Developer Process
## Standards

+ __Pull request and branches.__ Contribute by developing on a clone of the repo and writing the code in a branch. Submit a pull request when ready.

    For developers with push permission to the repo, see [Stan's process](https://github.com/stan-dev/stan/wiki/Developer-Process#information-to-include-in-pull-request) for how to name branches.  Do not merge your own pull requests or ever push to master. Someone should always review your code.  After merging (or deciding to close the request without merging), always delete the branch from the repo.

+ __Unit testing.__ Unit testing is awesome. It's useful not only for checking code after having written it, but also in checking code as you are developing it. If you're informally writing short scripts that output various things anyways, I suggest saving the file in the `tests/` directory. This gets the momentum going as the test becomes formalized. For testing, simply run [`nose2`](http://nose2.readthedocs.io/en/latest/getting_started.html) in the repo; it will automatically find and run all tests in `tests/`. Most if not all pull requests should have unit tests.
+ __Issue labeling system.__ We use [Stan's labeling system](https://github.com/stan-dev/stan/pulls). While several labels obviously don't apply to us, it's better than the default labels and it's not worth the effort to reinvent the wheel and maintain a custom system.

Coding

+ __Style guidelines.__ Follow [TensorFlow](https://www.tensorflow.org/versions/r0.9/how_tos/style_guide.html), including its [documentation guidelines](https://www.tensorflow.org/versions/r0.9/how_tos/documentation/index.html).  The only exceptions are detailed below. Some below are not necessarily exceptions but simply emphasize style guidelines you should be following from TensorFlow or PEP 8 anyways.
+ Use four-space indents rather than two-space.
+ To organize imports in a script, use three blocks: 1. any `from __future__ import [...]` lines; 2. any `import [...]` lines; and 3. any `from [...] import [...]` lines. Each block is separated by a blank line, and within each block the lines are sorted alphabetically.
+ `edward.stats` uses SciPy standards. This includes, for example, the argument specification and the choice of how a distribution is parameterized. `edward.models` uses `tf.contrib.distributions` standards.
+ For arguments that are positive integers, use `n_`, e.g., `n_minibatch`, `n_print`, to represent "number of [...]".  For class attributes that are booleans, use `is_`, e.g., `is_reparameterized`, `is_multivariate`.
+ Aim for 70 characters per line, with some exceptions.
+ [Package names are almost always plural, with the exception of](http://programmers.stackexchange.com/questions/75919/should-package-names-be-singular-or-plural) `util.py`.
+ Use a blank line to separate the end of an indented procedure:
```python
for i in range(5):
    do_stuff()

more_code()
```

## Suggested workflow

+ __Update your Python path.__ As you make changes, it can be cumbersome to constantly reinstall the package to test these changes. We recommend adding the path to your local repo to your `PYTHONPATH`. That is, run the following on the command-line:
```{bash}
export PYTHONPATH="${PYTHONPATH}:/path/to/repo"
```
For this to work permanently, add this line to your `~/.bashrc` if you use Bash or `~/.zshenv` if you use zsh. Any time you import the package, it will look for the package locally via this directory path.

+ __Local installation.__ Sometimes it does make sense to check the installation. To install locally, run the following when at the parent directory of the repo:
```{bash}
pip install -e edward
```
(We recommend not installing with `sudo`; rather [use virtualenv](http://docs.python-guide.org/en/latest/starting/install/osx/).)

+ __Packaging and submitting to PyPI.__ First, update the version number in `setup.py`. Second, follow [these steps](https://packaging.python.org/en/latest/distributing/#packaging-your-project). For shorthand, the sequence of commands is
```{bash}
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
Third, tag the release on Github and note the new additions when tagging this release. You can do this by comparing commits from the previous tagged release to master. A link that compares tagged commits to master is available on the [releases page](https://github.com/blei-lab/edward/releases).

## Suggested private workflow

To develop work on a branch privately, we suggest using a private repo that maintains the master branch from the public repo. Development happens on the private repo's branch, and when it is finished, you can push it to the public repo's branch to submit as a pull request. We describe this in detail.

Clone the private repo so you can work on it (create a repo if it does not exist).
```{bash}
git clone https://github.com/blei-lab/edward-private.git
```
Pull changes from the public repo. This will let the private repo have the latest code from the public repo on its master branch.
```{bash}
cd edward-private
git remote add public https://github.com/blei-lab/edward.git
git pull public master # Creates a merge commit
git push origin master
```
Now create your branch on the private repo, develop stuff, and pull any latest changes from the public repo as you develop (`git pull public master`). Make sure that as you're running Edward, you're using the Edward library pointing to the private repo so it reflects your developer changes and not pointed to the public repo where it won't see any changes.

Finally, to create a pull request from a private repo's branch to the public repo, push the private branch to the public repo.
```{bash}
git clone https://github.com/blei-lab/edward.git
cd edward
git remote add private https://github.com/blei-lab/edward-private.git
git checkout -b pull_request_yourname
git pull private master
git push origin pull_request_yourname
```
Now simply create a pull request via the Github UI on the public repo. Once project owners review the pull request, they can merge it. [Source](http://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private/30352360#30352360)
