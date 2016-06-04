# Documentation in Edward

Our documentation uses [MkDocs](http://mkdocs.org). It lets us build stand-alone pages for general documentation using just Markdown files. It also lets us auto-generate API documentation from the source code's docstrings.

## Building the documentation

+ Install MkDocs (`pip install mkdocs`).
+ You can build the documentation locally. Go to this `docs/` directory.
  Then run
```{bash}
mkdocs serve
```
+ We deploy the documentation so that it is available on this repo's
  Github pages.
  To do this (and assuming you have push permission), go to this
  `docs/` directory. Then run
```{bash}
mkdocs gh-deploy --clean
```
  We forward the main domain url to the Github
  pages url, [blei-lab.github.io/edward](http://blei-lab.github.io/edward). Following
  [Github's guide](https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages),
  namely, we make a `CNAME` file; then we update the DNS record on
  the DNS provider.

See [MkDocs](http://mkdocs.org) for more details such as how to
write documentation.

The theme we use is built off the source code in
[mkdocs-alabaster](https://github.com/iamale/mkdocs-alabaster).  There
were a number of non-customizable things we wanted to customize so we
copied the source and hacked at the html files directly.
