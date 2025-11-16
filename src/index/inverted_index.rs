use std::collections::HashMap;
type DocID = usize;

#[derive(Debug, Clone)]
pub struct PostingList {
    // Sorted document ids
    doc_ids: Vec<DocID>,
    //For the doc_id at position i, how many times does the term
    term_frequencies: Vec<usize>,
    // For the doc id at position i, where does it appear in the document
    positions: Vec<Vec<usize>>,
}

impl PostingList {
    pub fn new(
        doc_ids: Vec<DocID>,
        term_frequencies: Vec<usize>,
        positions: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            doc_ids,
            term_frequencies,
            positions,
        }
    }
}

impl Default for PostingList {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct InvertedIndex {
    postings: HashMap<String, PostingList>,
    doc_lengths: HashMap<DocID, usize>,
}

impl InvertedIndex {
    pub fn new(postings: HashMap<String, PostingList>, doc_lengths: HashMap<DocID, usize>) -> Self {
        Self {
            postings,
            doc_lengths,
        }
    }

    pub fn add_document(&mut self, doc_id: DocID, tokens: Vec<(String, usize)>) {
        let num_tokens = tokens.len();
        self.doc_lengths.insert(doc_id, num_tokens);
        for (token, pos) in tokens {
            match self.postings.get_mut(&token) {
                None => {
                    let posting = PostingList::new(vec![doc_id], vec![1], vec![vec![pos]]);
                    self.postings.insert(token, posting);
                }
                Some(posting) => {
                    let doc_id_pos = posting.doc_ids.binary_search(&doc_id);
                    match doc_id_pos {
                        Ok(doc_id_pos) => {
                            posting.term_frequencies[doc_id_pos] += 1;
                            posting.positions[doc_id_pos].push(pos);
                        }
                        Err(insert_pos) => {
                            posting.doc_ids.insert(insert_pos, doc_id);
                            posting.term_frequencies.insert(insert_pos, 1);
                            posting.positions.insert(insert_pos, vec![pos]);
                        }
                    }
                }
            }
        }
    }

    pub fn rank(&self, q: &str) -> Option<Vec<DocID>> {
        let posting = self.postings.get(q)?;
        let n = self.doc_lengths.len() as f64;
        let n_t = posting.doc_ids.len() as f64;
        let idf = (n / n_t).log(10.0);
        let mut doc_tf_idf: Vec<(DocID, f64)> = posting
            .doc_ids
            .iter()
            .zip(posting.term_frequencies.iter())
            .filter_map(|(doc_id, doc_term_n)| {
                let doc_len = *self.doc_lengths.get(doc_id)?;
                if doc_len == 0 {
                    return None;
                }

                let tf = (*doc_term_n as f64) / (doc_len as f64);

                Some((*doc_id, tf * idf))
            })
            .collect();

        doc_tf_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let res = doc_tf_idf.iter().map(|(doc_id, _)| *doc_id).collect();
        Some(res)
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new(HashMap::new(), HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn index_single_doc_single_term_single_position() {
        let mut index = InvertedIndex::default();

        index.add_document(1, vec![("hello".to_string(), 0)]);

        let posting = index
            .postings
            .get("hello")
            .expect("posting for 'hello' should exist");

        assert_eq!(posting.doc_ids, vec![1]);
        assert_eq!(posting.term_frequencies, vec![1]);
        assert_eq!(posting.positions, vec![vec![0]]);
        assert_eq!(index.doc_lengths.get(&1), Some(&1));
    }

    #[test]
    fn index_single_doc_single_term_multiple_positions() {
        let mut index = InvertedIndex::default();

        // same doc, same term, different positions
        index.add_document(
            1,
            vec![
                ("hello".to_string(), 0),
                ("hello".to_string(), 3),
                ("hello".to_string(), 5),
            ],
        );

        let posting = index
            .postings
            .get("hello")
            .expect("posting for 'hello' should exist");

        assert_eq!(posting.doc_ids, vec![1]);
        // three occurrences in the same doc
        assert_eq!(posting.term_frequencies, vec![3]);
        assert_eq!(posting.positions, vec![vec![0, 3, 5]]);
        assert_eq!(index.doc_lengths.get(&1), Some(&3));
    }

    #[test]
    fn index_single_term_multiple_docs_keeps_doc_ids_sorted() {
        let mut index = InvertedIndex::default();

        // Intentionally index docs out of order
        index.add_document(10, vec![("term".to_string(), 2)]);
        index.add_document(3, vec![("term".to_string(), 1)]);
        index.add_document(7, vec![("term".to_string(), 4)]);

        let posting = index
            .postings
            .get("term")
            .expect("posting for 'term' should exist");

        // doc_ids must be sorted because of binary_search + insert
        assert_eq!(posting.doc_ids, vec![3, 7, 10]);

        // each doc saw the term once
        assert_eq!(posting.term_frequencies, vec![1, 1, 1]);

        // each positions[i] should contain the single position we inserted
        assert_eq!(posting.positions, vec![vec![1], vec![4], vec![2]]);

        // doc lengths tracked for all docs
        assert_eq!(index.doc_lengths.get(&3), Some(&1));
        assert_eq!(index.doc_lengths.get(&7), Some(&1));
        assert_eq!(index.doc_lengths.get(&10), Some(&1));
    }

    #[test]
    fn index_multiple_terms_in_same_doc() {
        let mut index = InvertedIndex::default();

        index.add_document(
            1,
            vec![
                ("rust".to_string(), 0),
                ("search".to_string(), 1),
                ("rust".to_string(), 2),
            ],
        );

        // rust
        let rust_posting = index
            .postings
            .get("rust")
            .expect("posting for 'rust' should exist");

        assert_eq!(rust_posting.doc_ids, vec![1]);
        assert_eq!(rust_posting.term_frequencies, vec![2]);
        assert_eq!(rust_posting.positions, vec![vec![0, 2]]);

        // search
        let search_posting = index
            .postings
            .get("search")
            .expect("posting for 'search' should exist");

        assert_eq!(search_posting.doc_ids, vec![1]);
        assert_eq!(search_posting.term_frequencies, vec![1]);
        assert_eq!(search_posting.positions, vec![vec![1]]);

        // doc length
        assert_eq!(index.doc_lengths.get(&1), Some(&3));
    }

    #[test]
    fn index_same_doc_multiple_calls_accumulates_correctly() {
        let mut index = InvertedIndex::default();

        index.add_document(1, vec![("foo".to_string(), 0), ("foo".to_string(), 2)]);
        index.add_document(1, vec![("foo".to_string(), 5)]);

        let posting = index
            .postings
            .get("foo")
            .expect("posting for 'foo' should exist");

        assert_eq!(posting.doc_ids, vec![1]);
        assert_eq!(posting.term_frequencies, vec![3]);
        assert_eq!(posting.positions, vec![vec![0, 2, 5]]);

        // doc_lengths keeps the length of the *last* call for this doc_id
        assert_eq!(index.doc_lengths.get(&1), Some(&1));
    }

    #[test]
    fn default_creates_empty_index() {
        let index = InvertedIndex::default();
        assert!(index.postings.is_empty());
        assert!(index.doc_lengths.is_empty());
    }

    #[test]
    fn rank_returns_none_for_unknown_term() {
        let mut index = InvertedIndex::default();
        index.add_document(1, vec![("rust".to_string(), 0)]);

        let ranked = index.rank("unknown");
        assert!(ranked.is_none());
    }

    #[test]
    fn rank_orders_documents_by_tf_idf() {
        let mut index = InvertedIndex::default();

        // doc 1: "rust" once, length 1
        index.add_document(1, vec![("rust".to_string(), 0)]);

        // doc 2: "rust" twice, plus one extra token, length 3
        index.add_document(
            2,
            vec![
                ("rust".to_string(), 0),
                ("rust".to_string(), 1),
                ("extra".to_string(), 2),
            ],
        );

        // doc 3: does not contain "rust" (affects idf only)
        index.add_document(3, vec![("extra".to_string(), 0)]);

        let ranked = index.rank("rust").expect("should rank 'rust'");

        // doc 1 has tf = 1/1 = 1.0
        // doc 2 has tf = 2/3 < 1.0
        // both share the same idf, so doc 1 should come first
        assert_eq!(ranked, vec![1, 2]);
    }

    #[test]
    fn rank_skips_docs_without_length_info() {
        // Build an index where postings exist, but doc_lengths is empty.
        let mut postings = HashMap::new();
        postings.insert(
            "term".to_string(),
            PostingList::new(vec![1], vec![2], vec![vec![0, 1]]),
        );

        let index = InvertedIndex::new(postings, HashMap::new());

        // We still get Some, but the ranked list is empty because we can't
        // compute tf without a document length.
        let ranked = index.rank("term").expect("rank should return Some");
        assert!(ranked.is_empty());
    }
}
