.. _corpus:

Corpus Structure
=================

An Introduction to the :class:`~text_data.index.Corpus`
-------------------------------------------------------

In :ref:`getting_started`, I showed a brief tour into how you
can use :code:`text_data` to identify potential problems in your
analysis. Now, I want to go over how you can address those.

In addition to allowing you to enter a list of text documents,
:class:`~text_data.index.Corpus` objects allow you to enter tokenizers
when you initialize them. These tokenizers — so called because
they convert strings into a list of "tokens" — are fairly picky
in how they have to be initialized because of some of the search
features in :class:`~text_data.index.Corpus`. For full details,
see the :mod:`text_data.tokenize` module.

But the configuration you should generally be able to get away with
is illustrated in :func:`text_data.tokenize.corpus_tokenizer`.
This function accepts a regular expression pattern that will
split your text into a list of strings and a list of postprocessing
functions that will alter each item in that list of strings,
including by removing them. In our case, we noticed
that the default tokenizer :class:`~text_data.index.Corpus` uses,
which just splits on :code:`r"\w+"` regular expressions
kept onto a bunch of numbers that we didn't want. So let's change the
regular expression to only hold onto alphabetic words.

In addition there were a a few 1- or 2-letter words that didn't
really seem to convey much meaning and that felt to me like
they were possibly artifacts of bad tokenizing. (Specifically,
the default tokenizer will often handle apostrophes poorly.)
I'm going to address those by removing them from the data. If any
of your postprocessing functions returns :code:`None`,
:func:`text_data.tokenize.corpus_tokenizer` will simply remove them
from the final list of words.

So now, we're going to re-tokenize that original database, using a custom
tokenizer:

.. code-block::

    sotu_tokenizer = text_data.tokenize.corpus_tokenizer(
        r"[A-Za-z]+",
        [str.lower, lambda x: x if len(x) > 2 else None]
    )
    sotu_corpus = Corpus(list(sotu_data.speech), sotu_tokenizer)

Now, we can do the same thing we did before, looking at the TF-IDF
values across the corpus.

.. code-block::

    tfidf = sotu_corpus.tfidf_matrix()
    top_words, top_scores = sotu_corpus.get_top_words(tfidf, top_n=5)
    list(np.unique(top_words.flatten()))

There's still more tinkering that you could do — in a real project,
I might consider using WordNet, a tool that helps you reduce
words like "dogs" and "cats" into their root forms — but the results
I got look pretty decent, and are certainly better than what we had
before.

So with that in mind, I want to get started on the analysis task I have.
In particular, I want to see how Abraham Lincoln's speeches differed from his predecessor,
James Buchanan's. In order to do this, we're going to use two functions offered
by :code:`text_data` that help you morph your index into another index that's more
suitable for your analyis task. 

This is really useful in text analysis, because
you're often dealing with vague and changing definitions of what counts as a corpus.
Sometimes, you want to compare a document to all other documents in a corpus; sometimes,
you want to compare it to just one other document. And other times, as we're going
to do, you want to group a bunch of documents together and treat them as if they're a single
document.

We're going to use one function called :meth:`text_data.index.WordIndex.slice`
and another called :meth:`text_data.multi_corpus.flat_concat` to do this.
:meth:`text_data.index.WordIndex.slice` creates a new :class:`~text_data.index.Corpus`
object with the indexes we specify, while :meth:`text_data.multi_corpus.flat_concat`
combines and flattens a bunch of :class:`~text_data.index.Corpus` objects.

To start, let's find all of the speeches that either Obama or Bush gave:

.. code-block::

    lincoln = sotu_data[sotu_data.president == "Lincoln"]
    buchanan = sotu_data[sotu_data.president == "Buchanan"]

We could technically just instantiate these corpuses, much as we did to get
our entire corpus. But doing so would require tokenizing the corpuses again,
which would be slow. So let's instead create them using :meth:`text_data.index.WordIndex.slice`:

.. code-block::

    buchanan_corpus = sotu_corpus.slice(set(buchanan.index))
    lincoln_corpus = sotu_corpus.slice(set(lincoln.index))

And finally, let's combine these into a class called a :class:`~text_data.index.WordIndex`.
Essentially, this is the same thing as a :class:`~text_data.index.Corpus`, with the caveat
that we can't use the search functions I'll write about later.

.. code-block::

    both = text_data.multi_corpus.flat_concat(lincoln_corpus, buchanan_corpus)

Now, we can see what words distinguish Lincoln's State of the Union speeches from
Buchanan's.

To conduct the analysis, I'm going to use something called a log-odds ratio.
It's explained really well in `this paper <http://languagelog.ldc.upenn.edu/myl/Monroe.pdf>`_.
(That paper also conveys its limits; specifically, log-odds ratios do a poor job
of representing variance, but it's a decent metric for an introductory analysis.)

There's a bit more explanation of what a log-odds ratio is in the documentation
for :meth:`text_data.index.WordIndex.odds_word`. But making the computation itself
is easy:

.. code-block::

    log_odds = both.odds_matrix(sublinear=True)
    log_odds_ratio = log_odds[:,0] - log_odds[:,1]

And from there, we can visualize our findings by viewing the top 10 scoring
results from each candidate:

.. code-block::

    words, sorted_log_odds = both.get_top_words(log_odds_ratio)
    lincoln_words, top_lincoln = words[:10], sorted_log_odds[:10]
    buchanan_words, top_buchanan = words[-10:], sorted_log_odds[-10:]
    text_data.display.display_score_table(
        buchanan_words,
        top_buchanan,
        "Words Buchanan Used Disproportionately"
    )

.. raw:: html

    <p><b>Words Buchanan Used Disproportionately</b></p><table><thead><tr><th>Order</th><th>Word</th><th>Score</th></tr></thead><tbody><tr><td>1.</td><td>applied</td><td>-2.357841079138746</td></tr><tr><td>2.</td><td>conferred</td><td>-2.357841079138746</td></tr><tr><td>3.</td><td>silver</td><td>-2.357841079138746</td></tr><tr><td>4.</td><td>estimates</td><td>-2.4060753023089525</td></tr><tr><td>5.</td><td>company</td><td>-2.4060753023089525</td></tr><tr><td>6.</td><td>five</td><td>-2.4060753023089525</td></tr><tr><td>7.</td><td>employ</td><td>-2.4478979344312997</td></tr><tr><td>8.</td><td>whilst</td><td>-2.4847150241025275</td></tr><tr><td>9.</td><td>gold</td><td>-2.4847150241025275</td></tr><tr><td>10.</td><td>paraguay</td><td>-2.5738843074966002</td></tr></tbody></table>

.. code-block::

    text_data.display.display_score_table(
        lincoln_words,
        top_lincoln,
        "Words Lincoln Used Disproportionately"
    )

.. raw:: html

    <p><b>Words Lincoln Used Disproportionately</b></p><table><thead><tr><th>Order</th><th>Word</th><th>Score</th></tr></thead><tbody><tr><td>1.</td><td>emancipation</td><td>2.570928440185467</td></tr><tr><td>2.</td><td>space</td><td>2.4761184382773784</td></tr><tr><td>3.</td><td>agriculture</td><td>2.4465802463270254</td></tr><tr><td>4.</td><td>production</td><td>2.4137697411322385</td></tr><tr><td>5.</td><td>forward</td><td>2.335130804296506</td></tr><tr><td>6.</td><td>wages</td><td>2.335130804296506</td></tr><tr><td>7.</td><td>above</td><td>2.335130804296506</td></tr><tr><td>8.</td><td>run</td><td>2.2868970418386674</td></tr><tr><td>9.</td><td>propose</td><td>2.2868970418386674</td></tr><tr><td>10.</td><td>length</td><td>2.2868970418386674</td></tr></tbody></table>

You can see the difference between the two presidents immediately.
One of the words Buchanan uses disproportionately is "paraguay,"
likely a reference to Buchanan's attempt to annex Paraguay. Meanwhile,
one of Lincoln's most disproportionately used words is "emancipation,"
for obvious reasons.

But we can extend this analysis further by looking at bi-grams.
In natural language processing, a "bigram," is a two-word phrase
that's treated like a word.

Using :code:`text_data`, we can create indexes for any ngram we want
from within a :class:`Corpus` object. We can then access
the n-grams from within the corpus's :code:`ngram_indexes` attribute.

.. code-block::

    lincoln_corpus.add_ngram_index(n=2)
    buchanan_corpus.add_ngram_index(n=2)
    both_bigram = text_data.multi_corpus.flat_concat(
        lincoln_corpus.ngram_indexes[2],
        buchanan_corpus.ngram_indexes[2]
    )
    log_odds_bigram = both_bigram.odds_matrix(sublinear=True)
    log_odds_ratio_bigram = log_odds_bigram[:,0] - log_odds_bigram[:,1]
    bigrams, sorted_log_odds_bigram = both_bigram.get_top_words(log_odds_ratio_bigram)
    lincoln_bigrams, top_lincoln_bigrams = bigrams[:10], sorted_log_odds_bigram[:10]
    buchanan_bigrams, top_buchanan_bigrams = bigrams[-10:], sorted_log_odds_bigram[-10:]
    text_data.display.display_score_table(
        lincoln_bigrams,
        top_lincoln_bigrams,
        "Bigrams Lincoln Used Disproportionately"
    )

.. raw:: html

    <p><b>Bigrams Lincoln Used Disproportionately</b></p><table><thead><tr><th>Order</th><th>Word</th><th>Score</th></tr></thead><tbody><tr><td>1.</td><td>the measure</td><td>2.159969068846104</td></tr><tr><td>2.</td><td>free colored</td><td>2.159969068846104</td></tr><tr><td>3.</td><td>population and</td><td>2.159969068846104</td></tr><tr><td>4.</td><td>the railways</td><td>2.159969068846104</td></tr><tr><td>5.</td><td>which our</td><td>2.159969068846104</td></tr><tr><td>6.</td><td>the price</td><td>2.159969068846104</td></tr><tr><td>7.</td><td>the foreign</td><td>2.074724307333117</td></tr><tr><td>8.</td><td>agriculture the</td><td>2.074724307333117</td></tr><tr><td>9.</td><td>products and</td><td>2.074724307333117</td></tr><tr><td>10.</td><td>white labor</td><td>2.074724307333117</td></tr></tbody></table>

.. code-block::

    text_data.display.display_score_table(
        buchanan_bigrams,
        top_buchanan_bigrams,
        "Bigrams Buchanan Used Disproportionately"
    )

.. raw:: html

    <p><b>Bigrams Buchanan Used Disproportionately</b></p><table><thead><tr><th>Order</th><th>Word</th><th>Score</th></tr></thead><tbody><tr><td>1.</td><td>president and</td><td>-2.3591582598672822</td></tr><tr><td>2.</td><td>june the</td><td>-2.3591582598672822</td></tr><tr><td>3.</td><td>the ordinary</td><td>-2.3591582598672822</td></tr><tr><td>4.</td><td>three hundred</td><td>-2.407380077115034</td></tr><tr><td>5.</td><td>the capital</td><td>-2.4491916115755874</td></tr><tr><td>6.</td><td>hundred and</td><td>-2.4859986620392682</td></tr><tr><td>7.</td><td>present fiscal</td><td>-2.5188003424008407</td></tr><tr><td>8.</td><td>the constitutional</td><td>-2.548330416191556</td></tr><tr><td>9.</td><td>the island</td><td>-2.5751425427204833</td></tr><tr><td>10.</td><td>ending june</td><td>-2.6976458268603114</td></tr></tbody></table>

Now, we can clearly see the influence of the Civil War in the differences
between the two presidents' speeches, with Licoln clearly making repeated references
to the war.

Conclusion
----------

This illustrates how you can analyze text data to compare the language across
two sets of documents. :code:`text_data` offers a large number of tools for
concatenating and slicing data, making it easy to explore data
and compare the language used in a document set between different groups of people.

In the next section, I'll talk about how you can search through results
to get a better sense of the context in which certain language was used.