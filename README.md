# Text Data

This is a library to designed to help people explore and analyze
text data.

What does it do to achieve these goals? It:

- Uses an [inverted positional index](https://nlp.stanford.edu/IR-book/html/htmledition/positional-indexes-1.html) so you can efficiently
search through document sets.
- Has a minimal boolean querying and ranking system so you can search
for words or arbitrarily long phrases and have the documents that best
match your query appear at the top.
- Offers an extensive set of statistical lookups so you can get
the counts, frequencies, and even get a term-document matrix of
TF-IDF scores across a set of documents.

## Table of Contents

- [Pitch](#pitch)
- [Anti-Pitch](#anti-pitch)

## Pitch

When I analyzed text data for a story I wrote [on the way that politicians message themselves in different platforms](https://coloradosun.com/2020/09/04/cory-gardner-john-hickenlooper-campaign-messaging/),
I found myself spending an awfully large amount of the time exploring the data. This, of course, is standard in data journalism
and in data analysis more broadly. But I found that a lot of the work I was doing could be generalized to other projects.

In particular, I found myself wanting to run quick statistics on various words; wanting to figure out which words
made each candidate distinct; and wanting to view examples of each word.

This package is designed to make that exploration easy. 

## Anti-Pitch

This is not designed to deal with preprocessing or text cleaning at all. Both tasks are both a) too task-specific and
b) too well-handled by larger projects like `nltk` or `spacy` for it to be useful to incorporate in this work.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the 
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
