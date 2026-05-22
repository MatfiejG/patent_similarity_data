#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import multiprocessing
import os
import re
import sys
import timeit
from pathlib import Path

import gensim
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm


def increase_csv_field_size_limit():
    """
    Allow Python's CSV reader to handle long patent descriptions and claims.

    The default csv module field limit is too small for many full-text patent
    records. Some platforms cannot accept sys.maxsize directly, so the value is
    reduced until the interpreter accepts it.
    """
    field_size_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_size_limit)
            return
        except OverflowError:
            field_size_limit = int(field_size_limit / 10)


increase_csv_field_size_limit()

start = timeit.default_timer()

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

INPUT_CLAIMS_CSV = BASE_DIR / 'Data' / 'Raw' / 'all_patent_names_claims.csv'
INPUT_DESCRIPTIONS_CSV = BASE_DIR / 'Data' / 'Raw' / 'all_patent_names_description.csv'
#INPUT_CSV = BASE_DIR / 'Data' / 'Test' / 'all_patent_names_abstracts.csv'
OUTPUT_DIR = BASE_DIR / 'Outputs'
MODEL_PATH = OUTPUT_DIR / 'patent_doc2v_12e.model'
VECTORS_CSV_PATH = OUTPUT_DIR / 'patent_doc2v_vectors.csv'
VECTORS_JSONL_PATH = OUTPUT_DIR / 'patent_doc2v_vectors.jsonl'
SUMMARY_PATH = OUTPUT_DIR / 'patent_doc2v_summary.json'

CSV_DELIMITER = ';'
# The source files have used slightly different header names across exports.
# Each tuple is ordered from the preferred/current header to fallback names.
PATENT_ID_COLUMNS = ('patent_id', 'Patent_number', 'patent_number')
DESCRIPTION_TEXT_COLUMNS = ('Description', 'description', 'body', 'text')
CLAIMS_TEXT_COLUMNS = ('claims', 'Claims', 'claim', 'text')
MIN_WORDS = 200
VECTOR_SIZE = 300
EPOCHS = 12


def tokenize_text(text):
    """
    Convert a patent document into the token format expected by Doc2Vec.

    This keeps preprocessing intentionally light: line breaks are flattened,
    words are lowercased, and selected punctuation is kept as separate tokens.
    """
    text = text.replace('\n', ' ').strip()
    words = re.findall(r"[\w']+|[.,!?;]", text)
    return [word.lower() for word in words]


def first_populated_value(row, columns):
    """
    Return the first non-empty cell from a list of possible column names.

    This lets the script work with both the current raw CSV headers and older
    test/export headers without changing the rest of the pipeline.
    """
    for column in columns:
        candidate = row.get(column)
        if candidate:
            return candidate.strip()
    return ''


def iter_text_by_patent(csv_path, patent_id_columns, text_columns, delimiter):
    """
    Stream patent IDs and one selected text field from a CSV file.

    The function does not apply corpus-level filtering. It only normalizes the
    repeated task of finding the patent number and the best available text cell
    in a row.
    """
    with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        for row in reader:
            patent_id = first_populated_value(row, patent_id_columns)
            if not patent_id:
                continue

            yield patent_id, first_populated_value(row, text_columns)


def read_claims_by_patent(
    csv_path,
    patent_id_columns,
    text_columns,
    delimiter,
):
    """
    Load claims into a lookup keyed by patent ID.

    Descriptions are streamed later because they define the main iteration
    order. Claims are held in memory so each description row can be joined with
    its matching claims text in constant time.
    """
    claims_by_patent = {}

    for patent_id, claims in iter_text_by_patent(
        csv_path=csv_path,
        patent_id_columns=patent_id_columns,
        text_columns=text_columns,
        delimiter=delimiter,
    ):
        if patent_id not in claims_by_patent:
            claims_by_patent[patent_id] = claims

    return claims_by_patent


def iter_patent_records(
    descriptions_csv_path,
    claims_csv_path,
    patent_id_columns,
    description_text_columns,
    claims_text_columns,
    min_words,
    delimiter,
    progress_description=None,
):
    """
    Yield one training/export record per eligible patent.

    Each document is formed as description first and claims second. Patents are
    skipped when they have no identifier, are design patents, repeat a patent
    already seen in the descriptions file, or do not meet the minimum token
    threshold after the combined text is tokenized.
    """
    claims_by_patent = read_claims_by_patent(
        csv_path=claims_csv_path,
        patent_id_columns=patent_id_columns,
        text_columns=claims_text_columns,
        delimiter=delimiter,
    )
    seen_patents = set()

    description_records = iter_text_by_patent(
        csv_path=descriptions_csv_path,
        patent_id_columns=patent_id_columns,
        text_columns=description_text_columns,
        delimiter=delimiter,
    )
    if progress_description:
        description_records = tqdm(
            description_records,
            desc=progress_description,
            unit='row',
            dynamic_ncols=True,
        )

    for patent_id, description in description_records:
        if patent_id.upper().startswith('D'):
            continue
        if patent_id in seen_patents:
            continue

        claims = claims_by_patent.get(patent_id, '')
        # Preserve the requested document order: body/description, then claims.
        text_parts = [part for part in (description, claims) if part]
        text_value = '\n\n'.join(text_parts)

        words = tokenize_text(text_value)
        if len(words) < min_words:
            continue

        seen_patents.add(patent_id)
        yield {
            'patent_id': patent_id,
            'token_count': len(words),
            'words': words,
        }


class DocIterator(object):
    """
    Stream the corpus in the TaggedDocument format required by Gensim Doc2Vec.

    Gensim may iterate over this object more than once during training, so the
    class stores file paths and parsing settings instead of materializing the
    full patent corpus in memory.
    """

    def __init__(
        self,
        descriptions_csv_path,
        claims_csv_path,
        patent_id_columns,
        description_text_columns,
        claims_text_columns,
        min_words=200,
        delimiter=';',
    ):
        self.descriptions_csv_path = descriptions_csv_path
        self.claims_csv_path = claims_csv_path
        self.patent_id_columns = patent_id_columns
        self.description_text_columns = description_text_columns
        self.claims_text_columns = claims_text_columns
        self.min_words = min_words
        self.delimiter = delimiter

    def __iter__(self):
        for record in iter_patent_records(
            descriptions_csv_path=self.descriptions_csv_path,
            claims_csv_path=self.claims_csv_path,
            patent_id_columns=self.patent_id_columns,
            description_text_columns=self.description_text_columns,
            claims_text_columns=self.claims_text_columns,
            min_words=self.min_words,
            delimiter=self.delimiter,
            progress_description='Reading patents for Doc2Vec',
        ):
            yield TaggedDocument(record['words'], [record['patent_id']])


def export_vectors(
    model,
    descriptions_csv_path,
    claims_csv_path,
    patent_id_columns,
    description_text_columns,
    claims_text_columns,
    min_words,
    delimiter,
):
    """
    Export trained patent vectors and a small run summary.

    The vectors are written in two formats: CSV for dataframe/spreadsheet use
    and JSONL for line-oriented processing. The source corpus is rebuilt from
    the same description-plus-claims logic used during training, so token counts
    and exported patent IDs match the model's document tags.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(VECTORS_CSV_PATH, 'w', encoding='utf-8', newline='') as csv_outfile:
        csv_writer = csv.DictWriter(
            csv_outfile,
            fieldnames=['patent_id', 'token_count', 'vector_json'],
        )
        csv_writer.writeheader()

        with open(VECTORS_JSONL_PATH, 'w', encoding='utf-8') as jsonl_outfile:
            exported = 0

            for record in iter_patent_records(
                descriptions_csv_path=descriptions_csv_path,
                claims_csv_path=claims_csv_path,
                patent_id_columns=patent_id_columns,
                description_text_columns=description_text_columns,
                claims_text_columns=claims_text_columns,
                min_words=min_words,
                delimiter=delimiter,
                progress_description='Exporting vectors',
            ):
                patent_id = record['patent_id']
                vector = [float(value) for value in model.dv[patent_id]]
                vector_json = json.dumps(vector)

                csv_writer.writerow(
                    {
                        'patent_id': patent_id,
                        'token_count': record['token_count'],
                        'vector_json': vector_json,
                    }
                )
                jsonl_outfile.write(
                    json.dumps(
                        {
                            'patent_id': patent_id,
                            'token_count': record['token_count'],
                            'vector': vector,
                        }
                    )
                    + '\n'
                )
                exported += 1

    summary = {
        'input_descriptions_csv': str(descriptions_csv_path),
        'input_claims_csv': str(claims_csv_path),
        'vector_count': exported,
        'vector_size': model.vector_size,
        'epochs': model.epochs,
        'min_words': min_words,
        'patent_id_columns': list(patent_id_columns),
        'description_text_columns': list(description_text_columns),
        'claims_text_columns': list(claims_text_columns),
        'csv_delimiter': delimiter,
        'model_path': str(MODEL_PATH),
        'vectors_csv_path': str(VECTORS_CSV_PATH),
        'vectors_jsonl_path': str(VECTORS_JSONL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    # Fail early with a clear path when one of the required raw input files is
    # missing. Otherwise the error would appear later inside the training pass.
    if not INPUT_DESCRIPTIONS_CSV.exists():
        raise FileNotFoundError(
            'Input descriptions CSV not found: ' + str(INPUT_DESCRIPTIONS_CSV)
        )
    if not INPUT_CLAIMS_CSV.exists():
        raise FileNotFoundError('Input claims CSV not found: ' + str(INPUT_CLAIMS_CSV))

    workers = multiprocessing.cpu_count()
    # DocIterator streams joined patent documents directly into Doc2Vec.
    doc_iterator = DocIterator(
        descriptions_csv_path=INPUT_DESCRIPTIONS_CSV,
        claims_csv_path=INPUT_CLAIMS_CSV,
        patent_id_columns=PATENT_ID_COLUMNS,
        description_text_columns=DESCRIPTION_TEXT_COLUMNS,
        claims_text_columns=CLAIMS_TEXT_COLUMNS,
        min_words=MIN_WORDS,
        delimiter=CSV_DELIMITER,
    )

    model = gensim.models.Doc2Vec(
        documents=doc_iterator,
        workers=workers,
        vector_size=VECTOR_SIZE,
        epochs=EPOCHS,
    )

    # Older Gensim versions expose this cleanup method; newer versions do not.
    if hasattr(model, 'delete_temporary_training_data'):
        model.delete_temporary_training_data(
            keep_doctags_vectors=True,
            keep_inference=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    # Re-read the same joined corpus so the vector files include token counts
    # and only patents that passed the exact same filters as training.
    summary = export_vectors(
        model=model,
        descriptions_csv_path=INPUT_DESCRIPTIONS_CSV,
        claims_csv_path=INPUT_CLAIMS_CSV,
        patent_id_columns=PATENT_ID_COLUMNS,
        description_text_columns=DESCRIPTION_TEXT_COLUMNS,
        claims_text_columns=CLAIMS_TEXT_COLUMNS,
        min_words=MIN_WORDS,
        delimiter=CSV_DELIMITER,
    )

    print('Saved model to ' + str(MODEL_PATH))
    print('Saved vectors to ' + str(VECTORS_CSV_PATH))
    print('Saved JSONL vectors to ' + str(VECTORS_JSONL_PATH))
    print('Summary: ' + json.dumps(summary, indent=2))

    stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    print("Total running time: %d:%d:%d. \n" % (hours, mins, secs))
