//! Rust utilities for `text_data`.
//!
//! This library is designed to provide utilities for `text_data`
//! in Rust to improve performance. Functionally, this is largely
//! focused on a class `PositionalIndex` that creates a positional,
//! inverted index for use in `text_data`. This is a memory-efficient
//! and performant way of storing information about the words
//! and their frequencies. (Search engines often use these because
//! they are sparse but allow for quick access to the positions of
//! matching text.)
use genawaiter::stack::let_gen;
use genawaiter::yield_;
use ngrams::Ngrams;
use pyo3::class::mapping::PyMappingProtocol;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::iter::FromIterator;
use std::ops::Not;
/// A positional index, designed for cache-efficient performant searching.
///
/// This allows you to quickly search for phrases within a list of documents
/// and to compute basic calculations from the documents.
#[pyclass]
#[derive(Debug)]
struct PositionalIndex {
    /// Maps terms to their postings. Inspired from the
    /// [inverted_index](https://github.com/tikue/inverted_index/blob/master/src/index.rs) crate.
    index: BTreeMap<String, TermIndex>,
    /// The total number of *distinct* words in the document set
    #[pyo3(get)]
    vocab_size: usize,
    /// The total number of word occurrences in the document set
    #[pyo3(get)]
    num_words: usize,
    /// The total number of documents
    #[pyo3(get)]
    num_documents: usize,
}

impl Default for PositionalIndex {
    fn default() -> Self {
        PositionalIndex {
            index: BTreeMap::new(),
            vocab_size: 0,
            num_words: 0,
            num_documents: 0,
        }
    }
}

#[pymethods]
impl PositionalIndex {
    /// Creates the index.
    ///
    /// If documents is not specified, creates an empty (default) index.
    ///
    /// # Arguments
    ///
    /// * `documents`: An optional list of tokenized documents (i.e. a list of a list of words)
    /// * `indexes`: An optional list of the positions of the tokens in documents.
    ///
    /// # Errors
    ///
    /// Raises a ValueError if the documents and indexes don't have the same shape.
    #[new]
    fn new(
        documents: Option<Vec<Vec<String>>>,
        indexes: Option<Vec<Vec<(usize, usize)>>>,
    ) -> PyResult<Self> {
        let mut index = PositionalIndex::default();
        if let Some(docs) = documents {
            index.add_documents(docs, indexes)?;
        }
        Ok(index)
    }

    /// Updates the index with additional documents.
    ///
    /// # Arguments
    ///
    /// * `documents`: A list of tokenized documents (i.e. a list of a list of words)
    /// * `indexes`: An optional list of the positions of the tokens in documents.
    ///
    /// # Errors
    ///
    /// Raises a ValueError if the documents and indexes don't have the same shape.
    fn add_documents(
        &mut self,
        documents: Vec<Vec<String>>,
        indexes: Option<Vec<Vec<(usize, usize)>>>,
    ) -> PyResult<()> {
        if let Some(index_locations) = &indexes {
            if index_locations.len() != documents.len() {
                return Err(PyValueError::new_err(format!(
                    "The number of indexes does not match the number of documents. {} is not {}",
                    index_locations.len(),
                    documents.len()
                )));
            }
        }
        // need starting length so updates add documents with new ids
        let starting_id = self.num_documents;
        for (doc_index, words) in documents.iter().enumerate() {
            let document_id = doc_index + starting_id;
            if let Some(index_locations) = &indexes {
                // unwrap won't panic because of if let check above
                let num_indexes = index_locations.get(doc_index).unwrap().len();
                if num_indexes != words.len() {
                    return Err(PyValueError::new_err(format!(
                        "The number of indexes in document {} does not match the number of words: {} is not {}",
                        doc_index,
                        num_indexes,
                        words.len()
                    )));
                }
            }
            for (word_index, word) in words.iter().enumerate() {
                let word_position = match &indexes {
                    Some(index_locations) => {
                        let (start_idx, end_idx) = index_locations
                            .get(doc_index)
                            .unwrap()
                            .get(word_index)
                            .unwrap();
                        Position::new(word_index, Some(*start_idx), Some(*end_idx))
                    }
                    None => Position::new(word_index, None, None),
                };
                self.index
                    .entry(word.clone())
                    .or_insert(TermIndex::default())
                    .update(document_id, word_position);
                self.num_words += 1;
            }
        }
        self.vocab_size = self.index.len();
        self.num_documents += documents.len();
        Ok(())
    }

    /// Gets all of the words in the corpus.
    fn vocabulary(&self) -> HashSet<String> {
        HashSet::from_iter(self.index.keys().cloned())
    }

    /// Show the most commonly appearing words
    ///
    /// If num_words is left unspecified, returns all words.
    fn most_common(&self, num_words: Option<usize>) -> Vec<(String, usize)> {
        let mut word_counts = Vec::new();
        for (word, term_info) in self.index.iter() {
            word_counts.push((word.clone(), term_info.term_count));
        }
        word_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        word_counts
            .into_iter()
            // return the first n | vocab results
            .take(num_words.unwrap_or(self.vocab_size))
            .collect()
    }

    /// Count the total number of times a word has appeared
    ///
    /// This is based on `nltk.probability.FreqDist.count
    fn word_count(&self, word: &str) -> usize {
        self.index
            .get(word)
            .map(|v| v.term_count.clone())
            .unwrap_or(0)
    }

    /// Report the frequency in which a word appeared
    ///
    /// This is based on `nltk.probability.FreqDist.freq
    fn word_frequency(&self, word: &str) -> f64 {
        self.word_count(word) as f64 / self.num_words as f64
    }

    /// Count the number of documents a word has appeared in
    fn document_count(&self, word: &str) -> usize {
        self.index.get(word).map(|v| v.postings.len()).unwrap_or(0)
    }

    /// Report the document frequency (the percentage of documents that has the word)
    fn document_frequency(&self, word: &str) -> f64 {
        self.document_count(word) as f64 / self.num_documents as f64
    }

    /// Report the total number of times a word appears in a document
    fn term_count(&self, word: &str, document: usize) -> usize {
        self.index
            .get(word)
            .map(|v| v.postings.get(&document).map_or(0, |doc| doc.len()))
            .unwrap_or(0)
    }

    /// Searches for all of the documents where a given set of words appears
    ///
    /// This is not an exact phrase match; instead, it just finds
    /// examples where *one* of the words appears in the document.
    fn find_documents_with_words(&self, search_terms: Vec<String>) -> PyResult<HashSet<usize>> {
        Ok(self.get_matching_documents(&search_terms))
    }

    fn find_wordlist_positions(
        &self,
        search_terms: Vec<String>,
    ) -> HashSet<(usize, usize, usize, Option<usize>, Option<usize>)> {
        self.get_inexact_positions(&search_terms)
    }

    /// Finds all matching occurrences of a phrase.
    ///
    /// Includes information about the phrase's (document id,
    /// the starting index of the word, the starting index within the raw document,
    /// and the ending index within the raw document)
    fn find_phrase_positions(
        &self,
        search_terms: Vec<String>,
    ) -> PyResult<HashSet<(usize, usize, usize, Option<usize>, Option<usize>)>> {
        Ok(self.get_matching_phrases(&search_terms))
    }

    /// Finds all documents matching the exact phrase
    fn find_documents_with_phrase(&self, search_terms: Vec<String>) -> PyResult<HashSet<usize>> {
        let mut matches = HashSet::new();
        for match_item in self.get_matching_phrases(&search_terms) {
            // just add doc id
            matches.insert(match_item.0);
        }
        Ok(matches)
    }
}

#[pyproto]
impl PyMappingProtocol for PositionalIndex {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.num_documents)
    }
}

impl PositionalIndex {
    /// Finds all of the positions for documents containing the set of words.
    ///
    /// Returns results in form of (doc_id, first index, last index, raw start index, raw end index)
    fn get_inexact_positions(
        &self,
        search_terms: &Vec<String>,
    ) -> HashSet<(usize, usize, usize, Option<usize>, Option<usize>)> {
        let mut matches = HashSet::new();
        let matching_documents = self.get_matching_documents(search_terms);
        for document in matching_documents {
            for term in search_terms {
                let matching_term = self.index.get(term);
                if let Some(term_idx) = matching_term {
                    let matching_positions = term_idx.postings.get(&document);
                    if let Some(positions) = matching_positions {
                        for position in positions {
                            matches.insert((
                                document,
                                position.idx,
                                position.idx,
                                position.start_idx,
                                position.end_idx,
                            ));
                        }
                    }
                }
            }
        }
        matches
    }

    /// Creates an iterator of matching phrases
    ///
    /// Returns results in form of (doc_id, first index, last index, raw start index, raw end index)
    fn get_matching_phrases(
        &self,
        search_terms: &Vec<String>,
    ) -> HashSet<(usize, usize, usize, Option<usize>, Option<usize>)> {
        let matching_documents = self.get_matching_documents(search_terms);
        let search_order = self.get_search_terms(search_terms);
        let_gen!(matching_phrases, {
            // our phrase set starts off just being the full first set
            // then narrow it until it's empty or we're done
            if let Some(search_items) = search_order {
                if let Some((first_item, remaining_terms)) = search_items.split_first() {
                    let first_term = search_terms.get(*first_item).unwrap();
                    let first_postings = self.index.get(first_term).unwrap();
                    let last_postings = self
                        .index
                        .get(search_terms.get(*search_items.last().unwrap()).unwrap())
                        .unwrap();
                    let mut possible_values = self.get_formatted_postings(
                        first_postings,
                        &matching_documents,
                        first_item,
                    );
                    for term in remaining_terms {
                        let term_postings =
                            self.index.get(search_terms.get(*term).unwrap()).unwrap();
                        // // get_formatted_postings returns a HashSet of document ids
                        // // and the index of the first search term (assuming it's a match)
                        let term_values =
                            self.get_formatted_postings(term_postings, &matching_documents, term);
                        possible_values = possible_values
                            .intersection(&term_values)
                            .copied()
                            .collect();
                    }
                    for (doc_idx, first_index) in possible_values {
                        // values iterated over intersected (document, start_idx) values so unwrap fine
                        let first_positions = self
                            .index
                            .get(search_terms.first().unwrap())
                            .unwrap()
                            .postings
                            .get(&doc_idx)
                            .unwrap();
                        let last_positions = self
                            .index
                            .get(search_terms.last().unwrap())
                            .unwrap()
                            .postings
                            .get(&doc_idx)
                            .unwrap();
                        let start_idx = first_positions
                            .binary_search_by_key(&first_index, |a| a.idx)
                            .unwrap();
                        let last_index = &first_index + search_terms.len() - 1;
                        let last_idx = last_positions
                            .binary_search_by_key(&last_index, |a| a.idx)
                            .unwrap();
                        let full_tuple_val = (
                            doc_idx.clone(),
                            first_index.clone(),
                            last_index.clone(),
                            first_positions.get(start_idx).unwrap().start_idx,
                            last_positions.get(last_idx).unwrap().end_idx,
                        );
                        yield_!(full_tuple_val);
                    }
                }
            }
        });
        matching_phrases.into_iter().collect()
    }

    /// Gets a list of potential postings for a word given its term index and the set of matching documents
    ///
    /// This is a helper function for `get_matching_phrases` and produces the same output as that.
    /// (with the caveat that the start and end raw indexes are passed as tuple parameters instead of as
    /// hashset parameters)
    fn get_formatted_postings(
        &self,
        term_info: &TermIndex,
        documents: &HashSet<usize>,
        offset: &usize,
    ) -> HashSet<(usize, usize)> {
        let mut filtered_postings = HashSet::new();
        for document in documents {
            // this is a private helper for get_matching_phrases so unwrap should be fine
            let positions = term_info.postings.get(document).unwrap();
            for position in positions {
                // if the phrase is offset by a value larger than its position, it can't be valid
                if &position.idx >= offset {
                    // record start idx of whole phrase
                    // (this allows the phrase extractor to just intersect to get phrases)
                    filtered_postings.insert((document.clone(), position.idx - offset));
                }
            }
        }
        filtered_postings
    }

    /// Creates an iterator of documents that contain all of the search terms
    ///
    /// The search terms do not have to be in the correct order. But *every* search
    /// term must be in the document.
    fn get_matching_documents(&self, search_terms: &Vec<String>) -> HashSet<usize> {
        let search_order = self.get_search_terms(search_terms);
        let_gen!(matching_documents, {
            if let Some(search_items) = search_order {
                // because we're doing an all match, we just need to check
                // documents where the first item has the term.
                let term_split = search_items.split_first();
                if let Some((first_term, remaining_terms)) = term_split {
                    // unwrap is safe because of `get_search_terms` check
                    let first_term_documents = self
                        .index
                        .get(search_terms.get(*first_term).unwrap())
                        .unwrap()
                        .postings
                        .keys();
                    for document in first_term_documents {
                        let mut doc_in_all = true;
                        for other_term in remaining_terms {
                            let doc_has_term = self
                                .index
                                .get(search_terms.get(*other_term).unwrap())
                                .unwrap()
                                .postings
                                .contains_key(document);
                            if doc_has_term.not() {
                                doc_in_all = false;
                                break;
                            }
                        }
                        if doc_in_all {
                            yield_!(document.clone());
                        }
                    }
                }
            }
        });
        // TODO: Figure out how the hell to return an iterator
        HashSet::from_iter(matching_documents.into_iter())
    }

    /// Creates an iterator of search terms in the order in inverse term frequency order.
    ///
    /// Specifically, this returns a vector of the indexes in which the original search
    /// terms appear. This can improve the performance of searches.
    fn get_search_terms(&self, search_terms: &Vec<String>) -> Option<Vec<usize>> {
        let mut search_order = Vec::new();
        for (idx, term) in search_terms.iter().enumerate() {
            let term_freq = self.index.get(term).map(|v| v.term_count);
            if let Some(word_count) = term_freq {
                let insertion_spot = search_order
                    .binary_search(&(word_count, idx))
                    .unwrap_or_else(|x| x);
                search_order.insert(insertion_spot, (word_count, idx));
            } else {
                return None;
            }
        }
        Some(search_order.iter().map(|x| x.1).collect())
    }
}

/// Holds information about all of the postings for a single word (across documents)
#[derive(Debug)]
struct TermIndex {
    /// the frequency of the term across the documents
    pub term_count: usize,
    /// a mapping of documents to their postings
    pub postings: BTreeMap<usize, Vec<Position>>,
}

impl Default for TermIndex {
    fn default() -> Self {
        TermIndex {
            term_count: 0,
            postings: BTreeMap::new(),
        }
    }
}

impl TermIndex {
    /// Adds a new positional entry into the term index and updates statistics.
    pub fn update(&mut self, document_id: usize, position: Position) {
        self.term_count += 1;
        self.postings
            .entry(document_id)
            .or_insert(Vec::new())
            .push(position);
    }
}

/// Describes an individual position for a word within a document.usize
///
/// Functionally, this just stores information about the word's position within
/// a list of items and its start and end index within the original document
#[derive(Debug, Eq)]
struct Position {
    /// the index within the list of words
    pub idx: usize,
    /// the start index within the raw document
    pub start_idx: Option<usize>,
    /// The end index within the raw document
    pub end_idx: Option<usize>,
}

impl Position {
    /// Just a thin wrapper around the struct initialization.
    pub fn new(idx: usize, start_idx: Option<usize>, end_idx: Option<usize>) -> Position {
        Position {
            idx,
            start_idx,
            end_idx,
        }
    }
}

// impl ord so we can sort using binary search
impl Ord for Position {
    fn cmp(&self, other: &Self) -> Ordering {
        self.idx.cmp(&other.idx)
    }
}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

/// Takes a list of documents (where each document is a word) and builds n-grams from it.
///
/// Args:
///     documents: A list of documents where each document is a list of words
///     n: The n-gram size
///     sep: The separator between words. Defaults to a space.
///     prefix: The prefix before the first word (e.g. an opening tag) in each n-gram
///     suffix: The suffix after the last word (e.g. a closing tag) in each n-gram
#[pyfunction]
fn ngrams_from_documents(
    documents: Vec<Vec<String>>,
    n: Option<usize>,
    sep: Option<&str>,
    prefix: Option<String>,
    suffix: Option<String>,
) -> Vec<Vec<String>> {
    let mut ngram_docs = Vec::new();
    for document in documents {
        ngram_docs.push(ngram(
            document,
            n.clone(),
            sep.clone(),
            prefix.clone(),
            suffix.clone(),
        ))
    }
    ngram_docs
}
/// Takes a list of words and transforms it into a list of n-grams
///
/// # Arguments
/// * `document`: A list of words (representing a single document in PositionalIndex)
/// * `n`: The n-gram size. If unspecified, defaults to 1
/// * `sep`: The separator between words. Default is a space.
/// * `prefix`: The prefix before the first word (e.g. an opening tag)
/// * `suffix`: The suffix after the last word (e.g. a closing tag)
#[pyfunction]
fn ngram(
    document: Vec<String>,
    n: Option<usize>,
    sep: Option<&str>,
    prefix: Option<String>,
    suffix: Option<String>,
) -> Vec<String> {
    let gram_count = n.unwrap_or(1);
    let joiner = sep.unwrap_or(" ");
    let pre = prefix.unwrap_or(String::new());
    let post = suffix.unwrap_or(String::new());
    match gram_count {
        // technically, Ngrams works identically on empty vectors, but this allows x.first().unwrap()
        // to not panic.
        _ if document.len() == 0 => document,
        // for some reason Ngrams doesn't work well on documents with n=1
        0..=1 => document,
        // NGrams panics if the number of n-grams is larger than the size of the vector
        d if d > document.len() => vec![],
        _ => {
            Ngrams::new(document.into_iter(), gram_count)
                .map(|mut v| {
                    let first_or_none = v.first_mut();
                    // this adds a prefix + suffix to each element
                    if let Some(first_elem) = first_or_none {
                        *first_elem = pre.clone() + first_elem;
                        // last_mut is only None if first_mut is None
                        let last_elem = v.last_mut().unwrap();
                        *last_elem += &post;
                    }
                    v.join(&joiner)
                })
                .collect()
        }
    }
}

/// This module forms the Rust core of `text_data`.
#[pymodule]
fn text_data_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PositionalIndex>()?;
    m.add_function(wrap_pyfunction!(ngram, m)?)?;
    m.add_function(wrap_pyfunction!(ngrams_from_documents, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_order_is_inverse() {
        // `get_search_terms` should reveal search order in reverse
        let documents = vec![
            vec!["apple", "pie", "tastes", "good"],
            vec!["apple", "pie", "is", "good"],
            vec!["apple", "pie"],
        ]
        .iter()
        .map(|v| v.iter().map(|doc| doc.to_string()).collect())
        .collect();
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        let search_order = index.get_search_terms(&vec![
            "apple".to_string(),
            "is".to_string(),
            "good".to_string(),
        ]);
        assert!(search_order.is_some());
        assert_eq!(search_order.unwrap(), vec![1, 2, 0])
    }

    #[test]
    fn test_exact_phrase() {
        let documents = vec![
            vec!["apple", "pie", "is", "good"],
            vec!["i", "like", "apple", "pie"],
            vec!["no", "thanks"],
        ]
        .iter()
        .map(|v| v.iter().map(|doc| doc.to_string()).collect())
        .collect();
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        // make sure querying a word that's not in the vocabulary produces empty result
        let no_word_query = index.get_matching_phrases(&vec!["nonsense".to_string()]);
        assert!(no_word_query.is_empty());
        // querying a phrase that exists should work
        let matching_query =
            index.get_matching_phrases(&vec!["apple".to_string(), "pie".to_string()]);
        let expected = vec![(0, 0, None, None), (1, 2, None, None)];
        assert_eq!(matching_query, HashSet::from_iter(expected.iter().cloned()));
        // querying a set of documents that doesn't exist (but where words do) should be empty
        let reverse_order =
            index.get_matching_phrases(&vec!["pie".to_string(), "apple".to_string()]);
        assert!(reverse_order.is_empty());
    }

    #[test]
    fn test_ngram() {
        let sent = vec![
            "<s>".to_string(),
            "I".to_string(),
            "like".to_string(),
            "Rust".to_string(),
            "</s>".to_string(),
        ];
        let sent_bigram = ngram(
            sent,
            Some(2),
            Some("</w><w>"),
            Some("<w>".to_string()),
            Some("</w>".to_string()),
        );
        let expected = vec![
            "<w><s></w><w>I</w>".to_string(),
            "<w>I</w><w>like</w>".to_string(),
            "<w>like</w><w>Rust</w>".to_string(),
            "<w>Rust</w><w></s></w>".to_string(),
        ];
        assert_eq!(sent_bigram, expected);
        // empty vectors should produce empty vectors
        assert_eq!(ngram(vec![], None, None, None, None), Vec::<String>::new());
        // when there are no valid n-grams, it should also produce empty vectors
        assert_eq!(
            ngram(
                vec!["Happy".to_string(), "Birthday".to_string()],
                Some(5),
                None,
                None,
                None
            ),
            Vec::<String>::new()
        );
    }
}
