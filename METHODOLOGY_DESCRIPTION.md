# Methodological Description of the Patent Embedding Procedure

This document describes the analytical procedure implemented in `patent_d2v.py` in a form suitable for adaptation into a scholarly manuscript. It focuses on the conceptual logic of the code rather than on software installation or execution details.

## Overview

The procedure constructs distributed vector representations of patent documents using a Doc2Vec model trained on textual patent descriptions. The central aim is to transform heterogeneous patent text into fixed-length numerical vectors that preserve distributional information about document content. These vectors can subsequently be used as inputs for similarity measurement, clustering, exploratory analysis, or downstream statistical models of technological relatedness.

The implemented workflow follows four main stages. First, patent records are read from a structured text file containing patent identifiers and textual descriptions. Second, the text of each eligible patent is normalized and tokenized into a sequence of lowercased word and punctuation tokens. Third, the resulting document-token sequences are used to train a Doc2Vec model, which learns a dense vector representation for each patent. Fourth, the learned vectors are exported in tabular and line-delimited JSON formats to support subsequent analysis outside the training script.

## Input Data and Unit of Analysis

The unit of analysis is the individual patent. Each record is identified by a patent identifier and associated with a textual field, drawn from either an abstract column or a general text column. The procedure is designed to prioritize available patent text while maintaining a consistent document-level representation. If both candidate text fields are available, the first populated field according to the configured column order is used.

The procedure applies several eligibility rules before a patent enters the model corpus. Records without a patent identifier are excluded. Design patents, identified by patent identifiers beginning with the letter `D`, are excluded from the corpus. Duplicate patent identifiers are also removed, so that each patent contributes at most one document to the training data. Finally, documents shorter than a minimum token threshold are excluded to reduce the influence of extremely sparse textual records.

These filtering choices are intended to improve the interpretability and stability of the learned document representations. In particular, excluding very short records limits the inclusion of documents for which there is insufficient linguistic context to estimate a meaningful semantic position.

## Text Processing

Text preprocessing is deliberately lightweight. Each textual field is stripped of line breaks, normalized to lowercase, and tokenized into words and selected punctuation marks. This approach preserves the broad lexical content of the patent description while avoiding heavier linguistic transformations such as stemming, lemmatization, part-of-speech filtering, or stop-word removal.

The resulting representation treats each patent as an ordered sequence of tokens. This choice is consistent with the Doc2Vec framework, in which document vectors are learned from patterns of word co-occurrence and contextual usage across the corpus. The absence of aggressive preprocessing also helps preserve domain-specific terminology, abbreviations, and technical expressions that may be informative in patent text.

## Model Training

The model is trained using the Doc2Vec algorithm as implemented in the Gensim library. Doc2Vec extends distributional word embedding models by learning a vector representation for each document alongside word-level representations. In this workflow, each patent document is supplied to the model with its patent identifier as the document tag, allowing the trained model to associate each learned vector directly with the corresponding patent.

The configured model uses a 300-dimensional vector space and trains for 12 epochs. Parallel processing is enabled using the available CPU cores. The training corpus is streamed from the input file rather than fully materialized in memory, which makes the approach more suitable for large patent collections. During long corpus passes, terminal progress bars are displayed to provide visibility into the progress of data reading and vector export.

The resulting embedding space can be interpreted as a learned representation of textual relatedness among patents. Patents that use similar technical language or appear in similar lexical contexts are expected to occupy more proximate positions in the vector space. As with all distributional representations, the resulting similarities should be understood as text-based approximations of technological or semantic relatedness rather than direct measures of legal scope, inventive quality, or economic value.

## Vector Export and Reproducible Outputs

After training, the model is saved to disk and the learned document vectors are exported for external use. For each eligible patent, the procedure writes the patent identifier, the token count of the source document, and the corresponding vector representation. Vectors are exported in two complementary formats: a CSV file for spreadsheet- and dataframe-oriented workflows, and a JSONL file for record-wise processing in notebook or pipeline environments.

The procedure also writes a summary file containing key metadata about the run, including the input file path, the number of exported vectors, the vector dimensionality, the number of training epochs, the minimum document-length threshold, the text columns considered, the CSV delimiter, and the output paths. This metadata provides a compact record of the modeling configuration and supports reproducibility across analysis runs.

## Analytical Interpretation

The output of the procedure is a patent-level embedding matrix in which each row corresponds to a patent and each column corresponds to a learned latent dimension. These dimensions are not intended to have direct human-readable meanings individually. Instead, their analytical value lies in the geometry of the vector space as a whole. Distances, similarities, neighborhoods, and clusters in this space can be used to study relationships among patents based on their textual descriptions.

In empirical research, such embeddings may support analyses of technological proximity, novelty, knowledge recombination, or similarity among inventions. For example, cosine similarity between patent vectors can be used to operationalize text-based relatedness, while clustering methods can be applied to identify groups of patents with similar technical content. The exported vectors are intentionally stored in open formats to allow researchers to combine them with citation data, classification codes, inventor information, assignee data, or other patent-level covariates.

## Vector-Only Similarity Notebook

The notebook `patent_sim_data_vectors_only_notebook.ipynb` implements a downstream analysis stage that starts from the exported vector file rather than from the original patent text or a SQLite database. It reads `Outputs/patent_doc2v_vectors.csv`, checks that the expected patent identifier and JSON-encoded vector columns are present, drops incomplete records, parses each vector into numerical form, and stacks the resulting vectors into a NumPy matrix. The notebook also verifies that all loaded vectors have the same dimensionality before similarity calculations are performed.

After loading the vector matrix, the notebook creates an in-memory mapping from patent identifiers to vectors. This mapping supports a helper function, `patent_pair_sim(patent1, patent2)`, which returns the cosine similarity between two selected patent IDs when both vectors are available and nonzero. This provides a simple way to inspect individual patent-pair relationships before computing larger similarity outputs.

The main analytical operation in the notebook is the computation of all unique pairwise cosine similarities among the loaded patent vectors. The full cosine similarity matrix is calculated using scikit-learn, and only the upper-triangular off-diagonal entries are converted into a long-form table containing `patent_id_1`, `patent_id_2`, and `similarity`. This table represents each unordered patent pair once and can be used for ranking similar patents, filtering by similarity thresholds, or joining similarity scores to other patent-level data.

The pairwise similarity table is written to `Outputs/patent_pairwise_similarity.parquet` when Parquet support is available. If Parquet export fails, the notebook falls back to a compressed CSV file at `Outputs/patent_pairwise_similarity.csv.gz`. Finally, the notebook visualizes the empirical distribution of pairwise cosine similarities with a histogram and kernel density estimate, saving the plot as `Outputs/patent_similarity_distribution.png`.

This notebook should be understood as an exploratory, file-based replacement for legacy SQLite-dependent examples. Its all-pairs approach is straightforward and transparent, but it has quadratic computational and memory requirements because it constructs the full similarity matrix. For very large patent collections, the same conceptual procedure may need to be adapted to use batching, approximate nearest-neighbor methods, or similarity calculations restricted to analytically relevant subsets of patents.

## Methodological Considerations

Several methodological considerations should be noted when interpreting the resulting vectors. First, the learned representation depends on the textual fields available in the input data. If abstracts are used rather than full patent descriptions or claims, the embedding space reflects the information contained in those shorter summaries. Second, the exclusion of short documents and design patents shapes the population represented in the final vector set. Third, Doc2Vec models are stochastic and may vary across runs unless random seeds and the computational environment are explicitly controlled.

In this implementation, strict deterministic reproducibility is not enforced through a fixed random seed and single-threaded execution. This choice reflects a practical trade-off between computational reproducibility and computational feasibility. The patent corpus is potentially large, and training document embeddings over many epochs is computationally intensive. The procedure therefore uses multi-threaded training across the available CPU cores in order to make model estimation tractable within reasonable time constraints. Because parallel training can introduce small run-to-run differences in the order of model updates, the resulting vectors should be understood as realizations of the same stochastic embedding procedure rather than as bitwise-identical deterministic outputs.

Finally, the model captures textual similarity, not necessarily technological equivalence. Patents may be linguistically similar while differing in legal claims or commercial application, and technologically related patents may use different terminology across domains or time periods. The vectors should therefore be interpreted as one empirical representation of patent relatedness, preferably in combination with other indicators when used for substantive inference.
