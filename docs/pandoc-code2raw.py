#!/usr/bin/env python

"""Pandoc filter to insert arbitrary raw output markup
as Code/CodeBlocks with an attribute raw=<outputformat>.

Especially useful for inserting LaTeX code which pandoc will
otherwise mangle:

    ````{raw=latex}
    \let\Begin\begin
    \let\End\end
    ````
or for making HTML opaque to pandoc, which will otherwise
show the text between tags in other output formats,

or for allowing Markdown in the arguments of LaTeX commands
or the contents of LaTeX environments

    `\textsf{`{raw=latex}<span class=sans>San Seriffe</span>`}`{raw=latex}

    ````{raw=latex}
    \begin{center}
    ````
    This is *centered*!
    ````{raw=latex}
    \end{center}
    ````

or for header-includes in metadata:

    ---
    header-includes: |
      ````{raw=latex}
      \usepackage{pgfpages}
      \pgfpagesuselayout{2 on 1}[a4paper]
      ````
    ...

See <https://github.com/jgm/pandoc/issues/2139>
"""

from pandocfilters import RawInline, RawBlock, toJSONFilter
from pandocattributes import PandocAttributes

raw4code = {'Code': RawInline, 'CodeBlock': RawBlock}


def code2raw(key, val, format, meta):
    if key not in raw4code:
        return None
    attrs = PandocAttributes(val[0], format='pandoc')
    raw = attrs.kvs.get('raw', None)
    if raw:
        # if raw != format:     # but what if we output markdown?
        #     return []
        return raw4code[key](raw, val[-1])
    else:
        return None


if __name__ == "__main__":
    toJSONFilter(code2raw)
