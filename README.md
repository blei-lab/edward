# Edward website

The backend of our website depends on [pandoc](http://pandoc.org), [beautifulsoup](https://www.crummy.com/software/BeautifulSoup/), and [sphinx](http://www.sphinx-doc.org/). This lets us write stand-alone pages for documentation using LaTeX. It also lets us auto-generate API documentation from the source code's docstrings.

The frontend of our website depends on [skeleton.css](http://getskeleton.com/), [Google Fonts](https://www.google.com/fonts), [highlight.js](https://highlightjs.org/), and [KaTeX](https://khan.github.io/KaTeX/).

## Editing the website

All stand-alone pages are in `docs/tex`. These compile to root level HTML pages. Our custom pandoc HTML template is `docs/tex/edward_template.pandoc`.

## Building the website

+ Install the dependencies
```{bash}
pip install pandoc beautifulsoup4 sphinx sphinx-autobuild sphinx_rtd_theme ghp-import
```
+ You can build the website locally. Go to this `docs/` directory and run
```{bash}
./compile.sh
```

This will
  1. run pandoc on all LaTeX files in `docs/tex`
  2. run beautifulsoup4 on all output html files to clean up artifacts
  3. run sphinx's autobuild on `../sphinx` to auto-generate the API
  4. copy the API output into `docs/api`

## Deploying the website

+ We deploy the documentation so that it is available on this repo's
  Github pages (the `gh-pages` branch). To do this (and assuming you
  have push permission), go to this directory. Then run
```{bash}
./deploy.sh
```
  We forward the main domain url to the Github pages url,
  [blei-lab.github.io/edward](http://blei-lab.github.io/edward).
  Following
  [Github's guide](https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages),
  namely, we make a `CNAME` file; then we update the DNS record on
  the DNS provider.
