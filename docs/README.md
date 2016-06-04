# Documentation in Edward

Our documentation uses Markdown with [MkDocs](http://mkdocs.org). It also lets us auto-generate documentation from the source code's docstrings.

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
mkdocs gh-deploy
```
  We forward the main domain url to these Github
  pages, [blei-lab.github.io/edward](http://blei-lab.github.io/edward), following
  [Github's guide](https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages)
  (namely, we make a `CNAME` file in the `gh-pages` branch; then we
  update the DNS record on the DNS provider; this is the only
  file manually added to the `gh-pages` branch).

See [MkDocs](http://mkdocs.org) for more details, including how to
write documentation.

Regarding the theme, this is built off the source code in
[mkdocs-alabaster](https://github.com/iamale/mkdocs-alabaster).
However, there were a number of non-customizable things we wanted to
customize. So we copied the source and hacked at the html files. If
necessary, you can diff the files to note the changes.
