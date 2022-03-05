import nltk
import sys
import os
import string
import numpy as np


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files = os.listdir(directory)
    documents = {}
    for file in files:
        with open(os.path.join(directory, file), 'r', encoding="utf8") as f:
            documents[file] = f.read()

    return documents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    document_no_punc = document.translate(str.maketrans('', '', string.punctuation))
    tokenized_document = nltk.word_tokenize(document_no_punc.lower())
    tokenized_document_no_stop_words = [word for word in tokenized_document if word not in nltk.corpus.stopwords.words("english")]

    return tokenized_document_no_stop_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    idfs = {}
    tot_docs = len(documents)

    tot_word_list = []
    for word_list in documents.values():
        tot_word_list += word_list

    all_words = set(tot_word_list)

    for word in all_words:
        doc_count = 0
        for doc in documents.values():
            doc = set(doc)
            if word in doc:
                doc_count += 1
                continue
        idfs[word] = np.log(tot_docs / doc_count)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    top_files_dict = {}
    for file in files.keys():
        top_files_dict[file] = 0

    for word in query:
        for file_name, doc_words in files.items():
            word_count = doc_words.count(word)
            top_files_dict[file_name] += word_count * idfs[word]

    sorted_top_files = [key for key, val in sorted(top_files_dict.items(), key=lambda x: x[1], reverse=True)]

    return sorted_top_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    top_sentences_scoring_dict = {}

    for sentence, words in sentences.items():
        sentence_idf = 0
        words_filtered = [word for word in words if word in query]
        sentence_query_term_density = len(words_filtered) / len(words)
        for word in words_filtered:
            sentence_idf += idfs[word]
        top_sentences_scoring_dict[sentence] = (sentence_idf, sentence_query_term_density)

    sorted_top_sentences_qtd = {key: val for key, val in sorted(top_sentences_scoring_dict.items(), key=lambda x: x[1][1], reverse=True)}
    sorted_top_sentences_idf = [key for key, val in sorted(sorted_top_sentences_qtd.items(), key=lambda x: x[1][0], reverse=True)]

    return sorted_top_sentences_idf[:n]

if __name__ == "__main__":
    main()
