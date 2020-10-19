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
use ngrams::Ngrams;
use numpy::{PyArray1, PyArray2};
use pyo3::class::sequence::PySequenceProtocol;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter::FromIterator;
use std::ops::Not;

type PositionResult = (usize, usize, usize, Option<usize>, Option<usize>);
/// A positional index, designed for cache-efficient performant searching.
///
/// This allows you to quickly search for phrases within a list of documents
/// and to compute basic calculations from the documents.
#[pyclass]
#[derive(Debug, Clone)]
struct PositionalIndex {
    /// Maps terms to their postings. Inspired from the
    /// [inverted_index](https://github.com/tikue/inverted_index/blob/master/src/index.rs) crate.
    index: BTreeMap<String, TermIndex>,
    /// The total number of word occurrences in the document set
    #[pyo3(get)]
    num_words: usize,
    /// A vector marking the length of documents
    doc_lengths: BTreeMap<usize, usize>,
    /// The index after the last index in the document set
    next_idx: usize,
}

impl Default for PositionalIndex {
    fn default() -> Self {
        PositionalIndex {
            index: BTreeMap::new(),
            num_words: 0,
            doc_lengths: BTreeMap::new(),
            next_idx: 0,
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
        let starting_id = self.next_idx;
        let mut doc_lengths = BTreeMap::new();

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
            // keep track of the maximum index parsed to raise errors in car of overlapping indexes
            let mut max_index_parsed = None;
            for (word_index, word) in words.iter().enumerate() {
                let word_position = match &indexes {
                    Some(index_locations) => {
                        let (start_idx, end_idx) = index_locations
                            .get(doc_index)
                            .unwrap()
                            .get(word_index)
                            .unwrap();
                        if let Some(idx) = max_index_parsed {
                            if start_idx <= &idx {
                                return Err(PyValueError::new_err(
                                    "Positional indexes cannot overlap.".to_string(),
                                ));
                            }
                        }
                        if end_idx <= start_idx {
                            return Err(PyValueError::new_err(
                                "The starting and ending positions for an index must always increase.".to_string()
                            ));
                        }
                        max_index_parsed = Some(*end_idx);
                        Position::new(word_index, Some(*start_idx), Some(*end_idx))
                    }
                    None => Position::new(word_index, None, None),
                };
                self.index
                    .entry(word.clone())
                    .or_insert_with(TermIndex::default)
                    .update(document_id, word_position);
                self.num_words += 1;
            }
            doc_lengths.insert(document_id, words.len());
        }
        self.next_idx += documents.len();
        self.doc_lengths.extend(doc_lengths);
        Ok(())
    }

    // Filtering
    //
    // All of these functions are designed to create new indexes
    // based on previous indexes

    /// This copies the index
    fn copy(&self) -> PositionalIndex {
        self.clone()
    }

    /// Changes the index, having it iterate in order from the optional starting position (defaults at 0)
    fn reset_index(&mut self, start_index: Option<usize>) {
        let first_idx = start_index.unwrap_or(0);
        let mut base_index = PositionalIndex::default();
        // create a mapping from old indexes to new ones, while updating values
        let old_idx_to_new_idx: BTreeMap<usize, usize> = self
            .doc_lengths
            .iter()
            .enumerate()
            .map(|(count, res)| {
                let (doc_id, doc_len) = res;
                base_index.doc_lengths.insert(count + first_idx, *doc_len);
                (*doc_id, count + first_idx)
            })
            .collect();
        base_index.index.extend({
            self.index
                .par_iter()
                .map(|(word, term_idx)| {
                    let postings = term_idx
                        .postings
                        .iter()
                        .map(|(doc_id, positions)| {
                            (*old_idx_to_new_idx.get(doc_id).unwrap(), positions.to_vec())
                        })
                        .collect();
                    (
                        word.clone(),
                        TermIndex {
                            term_count: term_idx.term_count,
                            postings,
                        },
                    )
                })
                .collect::<BTreeMap<String, TermIndex>>()
        });
        base_index.next_idx = base_index.__len__();
        base_index.num_words = self.num_words;
        *self = base_index;
    }

    /// Converts a PositionalIndex with 1+ documents into a PositionalIndex with a single document
    fn flatten(&self) -> PositionalIndex {
        if self.__len__() == 0 {
            return self.clone();
        }
        let num_words = self.num_words;
        let mut doc_lengths = BTreeMap::new();
        doc_lengths.insert(0, self.doc_lengths.values().sum());
        let next_idx = 1;
        let index = self
            .index
            .par_iter()
            .map(|(word, term_idx)| {
                let mut postings = BTreeMap::new();
                // flatten the vector of positions, and remove positional info from it
                let term_items = term_idx
                    .postings
                    .values()
                    .map(|positions| positions.iter().map(|v| v.remove_positions()))
                    .flatten()
                    .collect();
                postings.insert(0, term_items);
                (
                    word.clone(),
                    TermIndex {
                        term_count: term_idx.term_count,
                        postings,
                    },
                )
            })
            .collect();
        PositionalIndex {
            index,
            num_words,
            doc_lengths,
            next_idx,
        }
    }

    /// Concatenates two indexes.
    ///
    /// If you set ignore_index=True, it will clear the first index
    /// (so the indexes will run from 0..combined length). Otherwise, you
    /// can get an error if the indexes are overlapping.
    #[staticmethod]
    fn concat(
        left_index: &PositionalIndex,
        right_index: &PositionalIndex,
        ignore_index: bool,
    ) -> PyResult<PositionalIndex> {
        if ignore_index {
            let mut res_index = left_index.clone();
            res_index.reset_index(None);
            let mut to_append = right_index.clone();
            to_append.reset_index(Some(left_index.__len__()));
            to_append.index.iter().for_each(|(word, term_idx)| {
                res_index
                    .index
                    .entry(word.clone())
                    .and_modify(|e| e.append(term_idx))
                    .or_insert_with(|| term_idx.clone());
            });
            res_index.num_words += to_append.num_words;
            res_index.doc_lengths.extend(to_append.doc_lengths);
            res_index.next_idx = res_index.__len__();
            Ok(res_index)
        } else {
            let left_idxs: HashSet<usize> = left_index.doc_lengths.keys().cloned().collect();
            let right_idxs: HashSet<usize> = right_index.doc_lengths.keys().cloned().collect();
            if left_idxs
                .intersection(&right_idxs)
                .cloned()
                .collect::<HashSet<usize>>()
                .is_empty()
                .not()
            {
                return Err(PyValueError::new_err("The two indexes for this function intersected. Hint: Try setting ignore_index=True.".to_string()));
            }
            let mut index_setup = left_index.clone();
            right_index.index.iter().for_each(|(word, term_idx)| {
                index_setup
                    .index
                    .entry(word.clone())
                    .and_modify(|e| e.append(term_idx))
                    .or_insert_with(|| term_idx.clone());
            });
            index_setup.index.extend(right_index.index.clone());
            index_setup
                .doc_lengths
                .extend(right_index.doc_lengths.clone());
            index_setup.num_words += right_index.num_words;
            index_setup.next_idx = right_index.next_idx;
            Ok(index_setup)
        }
    }

    /// This creates a new index, just from the values at the old index
    fn slice(&self, indexes: HashSet<usize>) -> PyResult<PositionalIndex> {
        let doc_lengths = indexes
            .iter()
            .map(|doc_id| {
                let doc_length = self.doc_lengths.get(doc_id).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Invalid ID. Document with ID {} does not exist",
                        doc_id
                    ))
                })?;
                Ok((*doc_id, *doc_length))
            })
            .collect::<PyResult<BTreeMap<usize, usize>>>()?;
        let num_words = doc_lengths.values().sum();
        let next_idx = doc_lengths.keys().max().unwrap_or(&0) + 1;
        let index = self
            .index
            .par_iter()
            .map(|(word, term_idx)| {
                let mut term_count = 0;
                let postings = term_idx
                    .postings
                    .iter()
                    .filter(|(doc_id, _positions)| doc_lengths.contains_key(doc_id))
                    .map(|(doc_id, positions)| {
                        term_count += positions.len();
                        (*doc_id, positions.clone())
                    })
                    .collect();
                (
                    word.clone(),
                    TermIndex {
                        term_count,
                        postings,
                    },
                )
            })
            .filter(|(_word, term_idx)| term_idx.term_count > 0)
            .collect();
        Ok(PositionalIndex {
            index,
            num_words,
            doc_lengths,
            next_idx,
        })
    }

    fn split_off(&mut self, indexes: HashSet<usize>) -> PyResult<PositionalIndex> {
        // this is inefficient, but i'm lazy and don't want to refactor
        let result_index = self.slice(indexes.clone())?;
        self.doc_lengths = self
            .doc_lengths
            .iter()
            .filter(|(doc_id, _num_words)| indexes.contains(doc_id).not())
            .map(|(doc_id, num_words)| (*doc_id, *num_words))
            .collect();
        self.next_idx = self.doc_lengths.keys().max().unwrap_or(&0) + 1;
        self.num_words = self.doc_lengths.values().sum();
        self.index = self
            .index
            .par_iter()
            .map(|(word, term_idx)| {
                let postings: BTreeMap<usize, Vec<Position>> = term_idx
                    .postings
                    .iter()
                    .filter(|(doc_id, _positions)| indexes.contains(doc_id).not())
                    .map(|(doc_id, positions)| (*doc_id, positions.clone()))
                    .collect();
                let term_count = postings.values().map(|v| v.len()).sum();
                (
                    word.clone(),
                    TermIndex {
                        postings,
                        term_count,
                    },
                )
            })
            .filter(|(_word, term_idx)| term_idx.term_count > 0)
            .collect();
        Ok(result_index)
    }

    /// This creates a new index that does not include the words
    /// of the original index (and that also doesn't include positional
    /// information)
    fn skip_words(&self, words: HashSet<String>) -> PositionalIndex {
        let next_idx = self.next_idx;
        let mut index = BTreeMap::new();
        let mut num_words = 0;
        let mut doc_lengths: BTreeMap<usize, usize> =
            self.doc_lengths.keys().map(|v| (*v, 0)).collect();
        for (word, term_idx) in &self.index {
            if words.contains(word).not() {
                let new_term_idx = term_idx.copy_without_positions();
                for (doc_id, posting_list) in new_term_idx.postings {
                    *doc_lengths.entry(doc_id).or_insert(0) += posting_list.len();
                }
                num_words += new_term_idx.term_count;
                index.insert(word.clone(), term_idx.copy_without_positions());
            }
        }
        PositionalIndex {
            next_idx,
            index,
            num_words,
            doc_lengths,
        }
    }

    // Corpus Information
    // These functions provide basic information about the corpus as a whole

    /// Gets all of the words in the corpus.
    fn vocabulary(&self) -> HashSet<String> {
        HashSet::from_iter(self.index.keys().cloned())
    }

    /// Gets all of the words in the corpus and returns them in sorted (list) order
    fn vocabulary_list(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Gets the number of unique words
    fn vocab_size(&self) -> usize {
        self.index.len()
    }

    // Point estimates
    // These functions provide point estimates for the index
    //
    // Word Statistics
    // These provide statistics about specific words
    // without requiring information about the document

    /// Count the number of documents a word has appeared in
    fn document_count(&self, word: &str) -> usize {
        self.index.get(word).map(|v| v.postings.len()).unwrap_or(0)
    }

    /// Report the document frequency (the percentage of documents that has the word)
    fn document_frequency(&self, word: &str) -> f64 {
        self.document_count(word) as f64 / self.doc_lengths.len() as f64
    }

    /// Get the inverse document frequency (the number of documents / the document count) for a word in a document
    fn idf(&self, word: &str) -> f64 {
        self.__len__() as f64 / self.document_count(word) as f64
    }

    /// Get a list of the documents that contain a word
    fn docs_with_word(&self, word: &str) -> Vec<usize> {
        self.index
            .get(word)
            .map(|v| v.postings.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get a mapping of the documents containing a word and its counts
    ///
    /// For a given word returns {doc_id: number of counts}
    fn word_counter(&self, word: &str) -> HashMap<usize, usize> {
        self.index
            .get(word)
            .unwrap_or(&TermIndex::default())
            .postings
            .par_iter()
            .map(|(doc_id, posting_dict)| (*doc_id, posting_dict.len()))
            .collect()
    }

    /// Show the most commonly appearing words
    ///
    /// If num_words is left unspecified, returns all words.
    fn most_common(&self, num_words: Option<usize>) -> Vec<(String, usize)> {
        let mut word_counts: Vec<(String, usize)> = self
            .index
            .par_iter()
            .map(|(word, term_info)| (word.clone(), term_info.term_count))
            .collect();
        word_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        word_counts
            .into_iter()
            // return the first n | vocab results
            .take(num_words.unwrap_or_else(|| self.index.len()))
            .collect()
    }

    /// Show the item that appears most often and its count
    fn max_word_count(&self) -> Option<(String, usize)> {
        self.most_common(Some(1)).first().map(|v| v.to_owned())
    }

    /// Count the total number of times a word has appeared
    ///
    /// This is based on `nltk.probability.FreqDist.count
    fn word_count(&self, word: &str) -> usize {
        self.index.get(word).map(|v| v.term_count).unwrap_or(0)
    }

    /// Report the frequency in which a word appeared
    ///
    /// This is based on `nltk.probability.FreqDist.freq
    fn word_frequency(&self, word: &str) -> f64 {
        self.word_count(word) as f64 / self.num_words as f64
    }

    /// Returns the odds of seeing a word at random.
    ///
    /// In other words, the frequency / (1 - frequency)
    fn odds_word(&self, word: &str, sublinear: bool) -> f64 {
        let word_freq = self.word_frequency(word);
        let odds = word_freq / (1. - word_freq);
        if sublinear.not() {
            odds
        } else {
            odds.log2()
        }
    }

    // Word-document point estimates
    //
    // These provide statistics about words within a document

    /// Report the total number of times a word appears in a document
    fn term_count(&self, word: &str, document: usize) -> PyResult<usize> {
        if document >= self.__len__() {
            return Err(PyValueError::new_err(format!(
                "Invalid document number. There are only {} documents in this index",
                document
            )));
        }
        let result = self
            .index
            .get(word)
            .map(|v| v.postings.get(&document).map_or(0, |doc| doc.len()))
            .unwrap_or(0);
        Ok(result)
    }

    /// Get the Raw term frequency (term count / document length) for a word in a document
    fn term_frequency(&self, word: &str, document: usize) -> PyResult<f64> {
        let term_count = self.term_count(word, document)?;
        Ok(term_count as f64 / *self.doc_lengths.get(&document).unwrap() as f64)
    }

    // Vector computations
    //
    // These functions return term-document vectors where each
    // item in the vector refers to a word in the vocabulary

    /// Get a numpy array of all the document counts in the corpus
    fn doc_count_vector(&self) -> Py<PyArray1<f64>> {
        self.term_matrix(|_word, term_idx| term_idx.postings.len() as f64)
    }

    /// Get a numpy array of all the document frequencies in the corpus
    fn doc_freq_vector(&self) -> Py<PyArray1<f64>> {
        self.term_matrix(|word, _term_idx| self.document_frequency(word))
    }

    /// Get a term matrix (|V| x 1) of all of the inverse document frequencies.
    fn idf_vector(&self) -> Py<PyArray1<f64>> {
        self.term_matrix(|word, _term_idx| self.idf(word))
    }

    /// Get a numpy array of the number of times words appeared in the corpus.
    fn word_count_vector(&self) -> Py<PyArray1<f64>> {
        self.term_matrix(|_word, term_idx| term_idx.term_count as f64)
    }

    /// Get a vector of the overall frequencies of words in the corpus
    fn word_freq_vector(&self) -> Py<PyArray1<f64>> {
        self.term_matrix(|word, _term_idx| self.word_frequency(word))
    }

    /// Get a vector of the odds or log-odds of seeing a word at random
    fn odds_vector(&self, sublinear: bool) -> Py<PyArray1<f64>> {
        self.term_matrix(|word, _term_idx| self.odds_word(word, sublinear))
    }

    // Matrix statistics

    /// Get a term-document matrix as a numpy array, where the values in the matrix are counts
    fn count_matrix(&self) -> Py<PyArray2<f64>> {
        self.tf_matrix(None, None)
    }

    /// Get a term-document matrix showing the term frequencies of all of the term,document pairs
    ///
    /// If `sublinear` is set to Some(true) or None, it translates the term frequencies into log space
    /// If `normalize` is set to Some(true), it returns the L1-norm of the raw count matrix
    /// (in other words, the counts normalized to the lengths of the documents)
    fn tf_matrix(&self, sublinear: Option<bool>, normalize: Option<bool>) -> Py<PyArray2<f64>> {
        let sublinear_tf = sublinear.unwrap_or(true);
        let l1_norm = normalize.unwrap_or(false);
        self.term_document_matrix(|word, _term_idx, doc_id| {
            let term_count = self.term_count(word, doc_id).unwrap() as f64;
            let adjusted_count = if sublinear_tf {
                term_count.log2() + 1.
            } else {
                term_count
            };
            if l1_norm {
                adjusted_count / *self.doc_lengths.get(&doc_id).unwrap() as f64
            } else {
                adjusted_count
            }
        })
    }

    /// Get a one-hot encoding matrix (showing all of the places where a given word exists/doesn't exist)
    fn one_hot_matrix(&self) -> Py<PyArray2<f64>> {
        self.term_document_matrix(|_w, term_idx, doc_id| {
            term_idx.postings.contains_key(&doc_id) as u8 as f64
        })
    }

    // Searching
    //
    // These are all designed to facilitate searches for the corpus

    /// Searches for all of the documents where a given set of words appears
    ///
    /// This is not an exact phrase match; instead, it just finds
    /// examples where *one* of the words appears in the document.
    fn find_documents_with_words(&self, search_terms: Vec<String>) -> HashSet<usize> {
        self.get_matching_documents(&search_terms)
    }

    fn find_wordlist_positions(&self, search_terms: Vec<String>) -> HashSet<PositionResult> {
        self.get_inexact_positions(&search_terms)
    }

    /// Finds all matching occurrences of a phrase.
    ///
    /// Includes information about the phrase's (document id,
    /// the starting index of the word, the starting index within the raw document,
    /// and the ending index within the raw document)
    fn find_phrase_positions(&self, search_terms: Vec<String>) -> HashSet<PositionResult> {
        self.get_matching_phrases(&search_terms)
    }

    /// Finds all documents matching the exact phrase
    fn find_documents_with_phrase(&self, search_terms: Vec<String>) -> HashSet<usize> {
        let mut matches = HashSet::new();
        for match_item in self.get_matching_phrases(&search_terms) {
            // just add doc id
            matches.insert(match_item.0);
        }
        matches
    }
}

#[pyproto]
impl PySequenceProtocol for PositionalIndex {
    fn __len__(&self) -> usize {
        self.doc_lengths.len()
    }

    fn __contains__(&self, item: &str) -> bool {
        self.index.contains_key(item)
    }
}

impl PositionalIndex {
    /// A helper function for creating term-document matrices (e.g. based on a word)
    fn term_document_matrix<F>(&self, func: F) -> Py<PyArray2<f64>>
    where
        F: std::ops::Fn(&str, &TermIndex, usize) -> f64 + std::marker::Sync,
    {
        Python::with_gil(|py| {
            let td_matrix: Vec<Vec<f64>> = self
                .index
                .par_iter()
                .map(|(word, term_idx)| {
                    // set a zeroed document
                    let mut zero_vec = vec![0.; self.__len__()];
                    term_idx.postings.keys().for_each(|doc_id| {
                        let doc_val = func(word, term_idx, *doc_id);
                        zero_vec[*doc_id] = doc_val;
                    });
                    zero_vec
                })
                .collect();
            PyArray2::from_vec2(py, &td_matrix).unwrap().to_owned()
        })
    }

    /// A helper function to create word-level matrices (computing statistics for all the words in the vocab)
    fn term_matrix<F>(&self, func: F) -> Py<PyArray1<f64>>
    where
        F: std::ops::Fn(&str, &TermIndex) -> f64 + std::marker::Sync,
    {
        Python::with_gil(|py| {
            let doc_count_vec: Vec<f64> = self
                .index
                .par_iter()
                .map(|(word, term_idx)| func(word, term_idx))
                .collect();
            PyArray1::from_vec(py, doc_count_vec).to_owned()
        })
    }
    /// Finds all of the positions for documents containing the set of words.
    ///
    /// Returns results in form of (doc_id, first index, last index, raw start index, raw end index)
    fn get_inexact_positions(&self, search_terms: &[String]) -> HashSet<PositionResult> {
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
    fn get_matching_phrases(&self, search_terms: &[String]) -> HashSet<PositionResult> {
        let mut matching_phrases = HashSet::new();
        let matching_documents = self.get_matching_documents(search_terms);
        let search_order = self.get_search_terms(search_terms);
        // our phrase set starts off just being the full first set
        // then narrow it until it's empty or we're done
        if let Some(search_items) = search_order {
            if let Some((first_item, remaining_terms)) = search_items.split_first() {
                let first_term = search_terms.get(*first_item).unwrap();
                let first_postings = self.index.get(first_term).unwrap();
                let mut possible_values =
                    self.get_formatted_postings(first_postings, &matching_documents, first_item);
                for term in remaining_terms {
                    let term_postings = self.index.get(search_terms.get(*term).unwrap()).unwrap();
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
                    let last_index = first_index + search_terms.len() - 1;
                    let last_idx = last_positions
                        .binary_search_by_key(&last_index, |a| a.idx)
                        .unwrap();
                    let full_tuple_val = (
                        doc_idx,
                        first_index,
                        last_index,
                        first_positions.get(start_idx).unwrap().start_idx,
                        last_positions.get(last_idx).unwrap().end_idx,
                    );
                    matching_phrases.insert(full_tuple_val);
                }
            }
        }
        matching_phrases
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
                    filtered_postings.insert((*document, position.idx - offset));
                }
            }
        }
        filtered_postings
    }

    /// Creates an iterator of documents that contain all of the search terms
    ///
    /// The search terms do not have to be in the correct order. But *every* search
    /// term must be in the document.
    fn get_matching_documents(&self, search_terms: &[String]) -> HashSet<usize> {
        let mut matching_documents = HashSet::new();
        let search_order = self.get_search_terms(search_terms);
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
                        matching_documents.insert(*document);
                    }
                }
            }
        }
        matching_documents
    }

    /// Creates an iterator of search terms in the order in inverse term frequency order.
    ///
    /// Specifically, this returns a vector of the indexes in which the original search
    /// terms appear. This can improve the performance of searches.
    fn get_search_terms(&self, search_terms: &[String]) -> Option<Vec<usize>> {
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
#[derive(Debug, Clone)]
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
            .or_insert_with(Vec::new)
            .push(position);
    }

    /// This creates a TermIndex that is identical minus positional
    /// information
    pub fn copy_without_positions(&self) -> TermIndex {
        let term_count = self.term_count;
        let mut postings = BTreeMap::new();
        for (doc_id, positions) in &self.postings {
            let cloned_postings = positions.iter().map(|v| v.remove_positions()).collect();
            postings.insert(*doc_id, cloned_postings);
        }
        TermIndex {
            term_count,
            postings,
        }
    }

    /// This updates a `TermIndex` with the contents of another
    pub fn append(&mut self, other: &TermIndex) {
        self.term_count += other.term_count;
        other.postings.iter().for_each(|(doc_id, postings)| {
            self.postings
                .entry(*doc_id)
                .and_modify(|e| {
                    e.extend(postings.clone());
                })
                .or_insert_with(|| postings.clone());
        })
    }
}

/// Describes an individual position for a word within a document.usize
///
/// Functionally, this just stores information about the word's position within
/// a list of items and its start and end index within the original document
#[derive(Debug, Eq, Clone, Copy)]
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

    /// This creates an identical position that does not contain
    /// positional information
    pub fn remove_positions(&self) -> Position {
        Position {
            idx: self.idx,
            start_idx: None,
            end_idx: None,
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
    documents
        .par_iter()
        .map(|document| ngram(document.to_vec(), n, sep, prefix.clone(), suffix.clone()))
        .collect()
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
    let pre = prefix.unwrap_or_default();
    let post = suffix.unwrap_or_default();
    match gram_count {
        // technically, Ngrams works identically on empty vectors, but this allows x.first().unwrap()
        // to not panic.
        _ if document.is_empty() => document,
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
#[cfg(not(tarpaulin_include))]
mod tests {
    use super::*;
    use counter::Counter;
    use proptest::prelude::*;

    fn generate_fake_documents() -> impl Strategy<Value = Vec<Vec<String>>> {
        prop::collection::vec(prop::collection::vec(".*", 0..20), 0..20)
    }

    fn build_counter(documents: &[Vec<String>]) -> Counter<String, usize> {
        // builds a Counter object from a set of documents
        let mut counter = Counter::new();
        for document in documents {
            let new_counter: Counter<String, usize> = document.iter().cloned().collect();
            counter += new_counter;
        }
        counter
    }

    #[test]
    fn adding_documents_passes() {
        // adding documents without an index should work
        let documents = vec![
            vec!["this", "is", "an", "example", "document"],
            vec!["document", "2", "document"],
        ]
        .iter()
        .map(|v| v.iter().map(|doc| doc.to_string()).collect())
        .collect();
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        assert_eq!(index.num_words, 8);
        assert_eq!(index.__len__(), 2);
        assert_eq!(index.index.len(), 6);
        let first_posting = Position {
            idx: 4,
            start_idx: None,
            end_idx: None,
        };
        let second_posting = Position {
            idx: 0,
            start_idx: None,
            end_idx: None,
        };
        let third_posting = Position {
            idx: 2,
            start_idx: None,
            end_idx: None,
        };
        let document_info = index.index.get("document").unwrap();
        assert_eq!(document_info.term_count, 3);
        assert_eq!(document_info.postings.len(), 2);
        assert_eq!(
            *document_info.postings.get(&0).unwrap().get(0).unwrap(),
            first_posting
        );
        assert_eq!(
            *document_info.postings.get(&1).unwrap().get(0).unwrap(),
            second_posting
        );
        assert_eq!(
            *document_info.postings.get(&1).unwrap().get(1).unwrap(),
            third_posting
        );
    }

    #[test]
    fn adding_document_with_positions_passes() {
        // if you add valid indexes (usize items of the same matrix size), it should pass
        let documents = vec![vec!["example".to_string(), "document".to_string()]];
        let positions = vec![vec![(0, 7), (11, 19)]];
        let result_index = PositionalIndex::new(Some(documents), Some(positions));
        assert!(result_index.is_ok());
        let index = result_index.unwrap();
        let example_posting = Position {
            idx: 0,
            start_idx: Some(0),
            end_idx: Some(7),
        };
        let document_posting = Position {
            idx: 1,
            start_idx: Some(11),
            end_idx: Some(19),
        };
        assert_eq!(
            *index
                .index
                .get("example")
                .unwrap()
                .postings
                .get(&0)
                .unwrap()
                .get(0)
                .unwrap(),
            example_posting
        );
        assert_eq!(
            *index
                .index
                .get("document")
                .unwrap()
                .postings
                .get(&0)
                .unwrap()
                .get(0)
                .unwrap(),
            document_posting
        );
    }

    #[test]
    fn invalid_indexes_raises_error_on_new() {
        // when you create an index with invalid indexes, it should raise an error
        let documents = vec![
            vec!["this".to_string(), "is".to_string()],
            vec![
                "a".to_string(),
                "set".to_string(),
                "of".to_string(),
                "documents".to_string(),
            ],
        ];
        let mismatch_len = vec![vec![(2, 19), (20, 77)]];
        // there should be as many documents in the index as there are in the document set
        let mismatched_len_index =
            PositionalIndex::new(Some(documents.clone()), Some(mismatch_len));
        assert!(mismatched_len_index.is_err());
        // each document should have the same number of indexes
        let mismatch_2 = vec![vec![(2, 19), (20, 77)], vec![(2, 19), (20, 77), (78, 99)]];
        let mismatch_2_index = PositionalIndex::new(Some(documents.clone()), Some(mismatch_2));
        assert!(mismatch_2_index.is_err());
        // the indexes shouldn't overlap
        let overlapping_indexes = vec![
            vec![(2, 19), (18, 77)],
            vec![(2, 19), (20, 77), (78, 99), (100, 105)],
        ];
        let overlap_index =
            PositionalIndex::new(Some(documents.clone()), Some(overlapping_indexes));
        assert!(overlap_index.is_err());
        let overlapping_2 = vec![
            vec![(2, 19), (20, 77)],
            vec![(2, 19), (20, 77), (78, 72), (100, 105)],
        ];
        let overlap_index_2 = PositionalIndex::new(Some(documents), Some(overlapping_2));
        assert!(overlap_index_2.is_err());
    }

    #[test]
    fn skipping_words_passes() {
        // makes sure that if you create an index that skips the words
        // the words are not in the new index
        let docs = vec![
            vec!["example".to_string(), "document".to_string()],
            vec!["document".to_string()],
        ];
        let basic_index = PositionalIndex::new(Some(docs.clone()), None).unwrap();
        let missing_words: HashSet<String> = vec!["document".to_string()].iter().cloned().collect();
        let skip_index = basic_index.skip_words(missing_words.clone());
        assert!(basic_index.__contains__("document"));
        assert!(skip_index.__contains__("document").not());
        assert_eq!(basic_index.__len__(), 2);
        // the second index should still have the same number of documents
        assert_eq!(skip_index.__len__(), 2);
        assert_eq!(basic_index.num_words, 3);
        assert_eq!(skip_index.num_words, 1);
        let expected_basic_lengths: BTreeMap<usize, usize> =
            vec![(0, 2), (1, 1)].iter().cloned().collect();
        assert_eq!(basic_index.doc_lengths, expected_basic_lengths);
        let expected_basic_lengths: BTreeMap<usize, usize> =
            vec![(0, 1), (1, 0)].iter().cloned().collect();
        assert_eq!(skip_index.doc_lengths, expected_basic_lengths);
        // skipping words should remove positional information
        let positions = vec![vec![(0, 2), (3, 5)], vec![(0, 5)]];
        let basic_positional = PositionalIndex::new(Some(docs), Some(positions)).unwrap();
        let ex_position = basic_positional
            .index
            .get("example")
            .unwrap()
            .postings
            .get(&0)
            .unwrap()
            .first()
            .unwrap();
        assert!(ex_position.start_idx.is_some());
        assert!(ex_position.end_idx.is_some());
        let skip_position = basic_positional.skip_words(missing_words);
        let example_skip = skip_position
            .index
            .get("example")
            .unwrap()
            .postings
            .get(&0)
            .unwrap()
            .first()
            .unwrap();
        assert!(example_skip.end_idx.is_none());
        assert!(example_skip.start_idx.is_none());
    }

    #[test]
    fn flattening_indexes_passes() {
        // flattening an index when there are no documents should return the same thing
        let blank_index = PositionalIndex::default();
        let flattened_blank = blank_index.flatten();
        assert_eq!(flattened_blank.__len__(), 0);
        assert_eq!(flattened_blank.num_words, 0);
        assert_eq!(flattened_blank.doc_lengths, BTreeMap::new());
        assert_eq!(flattened_blank.next_idx, 0);
        let new_docs = vec![
            vec!["example".to_string(), "document".to_string()],
            vec!["another".to_string(), "document".to_string()],
        ];
        let positions = vec![vec![(0, 1), (2, 5)], vec![(20, 29), (30, 34)]];
        let flattened_idx = PositionalIndex::new(Some(new_docs.clone()), Some(positions))
            .unwrap()
            .flatten();
        assert_eq!(flattened_idx.__len__(), 1);
        assert_eq!(flattened_idx.num_words, 4);
        assert_eq!(
            flattened_idx.doc_lengths,
            vec![(0, 4)].iter().cloned().collect()
        );
        assert_eq!(flattened_idx.next_idx, 1);
        for pos in flattened_idx
            .index
            .get("document")
            .unwrap()
            .postings
            .get(&0)
            .unwrap()
        {
            assert!(pos.start_idx.is_none());
            assert!(pos.end_idx.is_none());
        }
    }

    #[test]
    fn concatenate_full() {
        // if you don't set ignore_index to concat, it can throw an error
        let docs1 = vec![
            vec!["example".to_string(), "document".to_string()],
            vec!["another".to_string(), "example".to_string()],
        ];
        let docs2 = vec![vec![
            "a".to_string(),
            "new".to_string(),
            "document".to_string(),
            "set".to_string(),
        ]];
        let index1 = PositionalIndex::new(Some(docs1), None).unwrap();
        let index2 = PositionalIndex::new(Some(docs2), None).unwrap();
        assert!(PositionalIndex::concat(&index1, &index2, false).is_err());
        assert!(PositionalIndex::concat(&index1, &index2, true).is_ok());
        let safe_concat = PositionalIndex::concat(&index1, &index2, true).unwrap();
        assert_eq!(
            safe_concat.clone().most_common(None),
            vec![
                ("document".to_string(), 2),
                ("example".to_string(), 2),
                ("a".to_string(), 1),
                ("another".to_string(), 1),
                ("new".to_string(), 1),
                ("set".to_string(), 1)
            ]
        );
        assert_eq!(safe_concat.clone().vocab_size(), 6);
        let document_data = safe_concat.index.get("document");
        assert!(document_data.is_some());
        assert_eq!(document_data.unwrap().term_count, 2);
        assert_eq!(document_data.unwrap().postings.len(), 2);
        assert_eq!(
            *document_data.unwrap().postings.get(&0).unwrap(),
            vec![Position::new(1, None, None)]
        );
        assert!(document_data.unwrap().postings.get(&1).is_none());
        assert_eq!(
            *document_data.unwrap().postings.get(&2).unwrap(),
            vec![Position::new(2, None, None)]
        );
    }

    #[test]
    fn test_slice_split_off() {
        // this tests the two functions designed to split off indices work
        let docs = vec![
            vec!["just".to_string(), "going".to_string()],
            vec!["split".to_string()],
            vec!["parts".to_string()],
            vec!["of".to_string(), "this".to_string(), "off".to_string()],
        ];
        let mut index = PositionalIndex::new(Some(docs), None).unwrap();
        assert!(index.slice([0, 10].iter().cloned().collect()).is_err());
        let sliced = index.slice([0, 2].iter().cloned().collect()).unwrap();
        // make sure this hasn't removed items yet
        assert_eq!(index.__len__(), 4);
        assert_eq!(sliced.__len__(), 2);
        assert_eq!(sliced.num_words, 3);
        assert_eq!(
            sliced.doc_lengths,
            [(0, 2), (2, 1)].iter().cloned().collect()
        );
        assert_eq!(sliced.next_idx, 3);
        assert_eq!(
            sliced.most_common(None),
            vec![
                ("going".to_string(), 1),
                ("just".to_string(), 1),
                ("parts".to_string(), 1)
            ]
        );
        assert!(index.split_off([0, 10].iter().cloned().collect()).is_err());
        assert_eq!(index.__len__(), 4);
        index.split_off([0, 2].iter().cloned().collect()).unwrap();
        assert_eq!(index.__len__(), 2);
        assert_eq!(index.num_words, 4);
        assert_eq!(
            index.doc_lengths,
            [(1, 1), (3, 3)].iter().cloned().collect()
        );
        assert_eq!(index.next_idx, 4);
    }

    #[test]
    fn test_reindex() {
        let docs = vec![
            vec!["example".to_string()],
            vec!["document".to_string()],
            vec!["another".to_string()],
            vec!["document".to_string()],
        ];
        let index = PositionalIndex::new(Some(docs), None).unwrap();
        let mut sliced = index.slice([1, 3].iter().cloned().collect()).unwrap();
        assert_eq!(
            sliced.doc_lengths.keys().cloned().collect::<Vec<usize>>(),
            vec![1, 3]
        );
        sliced.reset_index(None);
        assert_eq!(
            sliced.doc_lengths.keys().cloned().collect::<Vec<usize>>(),
            vec![0, 1]
        );
        sliced.reset_index(Some(10));
        assert_eq!(
            sliced.doc_lengths.keys().cloned().collect::<Vec<usize>>(),
            vec![10, 11]
        );
    }

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
        let search_order =
            index.get_search_terms(&["apple".to_string(), "is".to_string(), "good".to_string()]);
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
        let no_word_query = index.get_matching_phrases(&["nonsense".to_string()]);
        assert!(no_word_query.is_empty());
        // querying a phrase that exists should work
        let matching_query = index.get_matching_phrases(&["apple".to_string(), "pie".to_string()]);
        let expected = vec![(0, 0, 1, None, None), (1, 2, 3, None, None)];
        assert_eq!(matching_query, HashSet::from_iter(expected.iter().cloned()));
        // querying a set of documents that doesn't exist (but where words do) should be empty
        let reverse_order = index.get_matching_phrases(&["pie".to_string(), "apple".to_string()]);
        assert!(reverse_order.is_empty());
    }

    #[test]
    fn searching_documents_passes() {
        // makes sure that searching for documents containing all items in a list of words works
        let documents = vec![
            vec!["the".to_string(), "final".to_string(), "word".to_string()],
            vec![
                "the".to_string(),
                "final".to_string(),
                "countdown".to_string(),
            ],
            vec!["finally".to_string()],
        ];
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        // empty search should produce empty set
        assert_eq!(index.get_matching_documents(&[]), HashSet::new());
        let mut the_final = vec!["the".to_string(), "final".to_string()];
        let final_search = index.get_matching_documents(&the_final);
        let expected_res = [0, 1].iter().cloned().collect();
        // 0, 1 have the words "the" and "final"
        assert_eq!(final_search, expected_res);
        let reversed_final: Vec<String> = the_final.iter().cloned().rev().collect();
        // the order of the search shouldn't matter for this.
        assert_eq!(index.get_matching_documents(&reversed_final), expected_res);
        // *all* words must appear in the set of resulting documents (not just 2 of the three)
        the_final.push("countdown".to_string());
        assert_eq!(
            index.get_matching_documents(&the_final),
            [1].iter().cloned().collect()
        );
        // if there aren't any documents matching the set of terms, should get an empty set
        the_final.push("word".to_string());
        assert_eq!(index.get_matching_documents(&the_final), HashSet::new());
    }

    #[test]
    fn test_inexact_positions() {
        // searching for the positions of documents containing every word in a phrase should work
        let documents = vec![
            vec!["the".to_string(), "final".to_string(), "word".to_string()],
            vec!["final".to_string(), "in".to_string(), "the".to_string()],
            vec!["final".to_string()],
        ];
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        let final_matches = index.get_inexact_positions(&["the".to_string(), "final".to_string()]);
        let expected = [
            (0, 0, 0, None, None),
            (0, 1, 1, None, None),
            (1, 0, 0, None, None),
            (1, 2, 2, None, None),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(final_matches, expected);
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

    #[test]
    fn test_vocabulary() {
        // getting the vocabulary from a positional index should work
        let mut index = PositionalIndex::new(None, None).unwrap();
        assert_eq!(index.vocabulary(), HashSet::new());
        let new_docs = vec![
            vec![
                "these".to_string(),
                "are".to_string(),
                "sample".to_string(),
                "documents".to_string(),
            ],
            vec![
                "this".to_string(),
                "is".to_string(),
                "another".to_string(),
                "document".to_string(),
            ],
            vec!["sample".to_string(), "document".to_string()],
        ];
        index.add_documents(new_docs, None).unwrap();
        let expected = vec![
            "these".to_string(),
            "are".to_string(),
            "sample".to_string(),
            "documents".to_string(),
            "this".to_string(),
            "is".to_string(),
            "another".to_string(),
            "document".to_string(),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(index.vocabulary(), expected);
    }

    #[test]
    fn test_most_common() {
        let documents = vec![
            vec!["is".to_string(), "a".to_string(), "a".to_string()],
            vec!["a".to_string(), "or".to_string(), "is".to_string()],
        ];
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        let expected = vec![
            ("a".to_string(), 3),
            ("is".to_string(), 2),
            ("or".to_string(), 1),
        ];
        assert_eq!(index.most_common(None), expected);
        assert_eq!(index.most_common(Some(1)), vec![("a".to_string(), 3)]);
        assert_eq!(
            index.most_common(Some(2)),
            vec![("a".to_string(), 3), ("is".to_string(), 2)]
        );
        assert_eq!(index.most_common(Some(3)), expected);
        assert_eq!(index.most_common(Some(1000)), expected);
    }

    #[test]
    fn test_word_count_invalid_is_0() {
        // if you try to get the count of a word that isn't in the vocabulary, you should get 0
        let index = PositionalIndex::new(None, None).unwrap();
        assert_eq!(index.word_count("nonsense"), 0);
    }

    #[test]
    fn test_document_count() {
        // makes sure getting the number of documents a word appeared in works
        let documents = vec![
            vec!["the".to_string(), "example".to_string()],
            vec!["the".to_string(), "document".to_string()],
            vec!["example".to_string()],
        ];
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        assert_eq!(index.document_count("the"), 2);
        assert_eq!(index.document_count("example"), 2);
        assert_eq!(index.document_count("document"), 1);
        assert_eq!(index.document_count("nonsense"), 0);
    }

    #[test]
    fn test_term_count() {
        // makes sure getting the total number of times a word appears in a document works
        let documents = vec![
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "a".to_string(),
            ],
            vec!["b".to_string(), "a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];
        let index = PositionalIndex::new(Some(documents), None).unwrap();
        // you must enter a valid document index
        assert!(index.term_count("a", 100).is_err());
        assert_eq!(index.term_count("a", 0).unwrap(), 3);
        assert_eq!(index.term_count("a", 1).unwrap(), 1);
        assert_eq!(index.term_count("a", 2).unwrap(), 0);
        assert_eq!(index.term_count("b", 0).unwrap(), 1);
        assert_eq!(index.term_count("b", 1).unwrap(), 2);
        assert_eq!(index.term_count("b", 2).unwrap(), 0);
        assert_eq!(index.term_count("c", 0).unwrap(), 0);
        assert_eq!(index.term_count("c", 1).unwrap(), 0);
        assert_eq!(index.term_count("c", 2).unwrap(), 1);
    }

    proptest! {

        #[test]
        fn test_word_count_existing(documents in generate_fake_documents()) {
            // makes sure getting the total word count for words in the vocabulary works
            let counter = build_counter(&documents);
            let index = PositionalIndex::new(Some(documents), None).unwrap();
            for (word, count) in counter.iter() {
                assert_eq!(index.word_count(&word), *count);
            }
        }

        #[test]
        fn test_num_words(documents in generate_fake_documents()) {
            // makes sure the number of words counted by `PositionalIndex.num_words` is correct
            let counter = build_counter(&documents);
            let expected = counter.values().sum();
            let index = PositionalIndex::new(Some(documents), None).unwrap();
            assert_eq!(index.num_words, expected);
        }

        #[test]
        fn test_vocab_size(documents in generate_fake_documents()) {
            // makes sure `PositionalIndex.vocab_size` works
            let counter = build_counter(&documents);
            let index = PositionalIndex::new(Some(documents), None).unwrap();
            assert_eq!(index.index.len(), counter.keys().len());
        }

        #[test]
        fn concatenating_words(
            documents_1 in generate_fake_documents(),
            documents_2 in generate_fake_documents()
        ) {
            let left_idx = PositionalIndex::new(Some(documents_1), None).unwrap();
            let right_idx = PositionalIndex::new(Some(documents_2), None).unwrap();
            let concat_index = PositionalIndex::concat(
                &left_idx,
                &right_idx,
                true
            ).unwrap();
            assert_eq!(left_idx.__len__() + right_idx.__len__(), concat_index.__len__());
            assert_eq!(
                left_idx.vocabulary().union(&right_idx.vocabulary()).cloned().collect::<HashSet<String>>(),
                concat_index.vocabulary()
            );
            assert_eq!(concat_index.next_idx, concat_index.__len__());
        }
    }
}
