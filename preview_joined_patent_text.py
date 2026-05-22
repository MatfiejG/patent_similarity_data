#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preview how patent descriptions and claims are joined before Doc2Vec training.

This script is intentionally small and read-only. It does not train a model and
does not write outputs. By default, it picks one random patent from the
description file, finds matching claims, and prints:

- patent number
- description/body text
- claims text
- combined text in the same order used by patent_algo_d2v.py
"""

import argparse
import random

from patent_algo_d2v import (
    CLAIMS_TEXT_COLUMNS,
    CSV_DELIMITER,
    DESCRIPTION_TEXT_COLUMNS,
    INPUT_CLAIMS_CSV,
    INPUT_DESCRIPTIONS_CSV,
    PATENT_ID_COLUMNS,
    iter_text_by_patent,
)


def shorten_text(text, max_chars):
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + '\n\n[... truncated ...]'


def read_claims_lookup():
    claims_by_patent = {}

    for patent_id, claims in iter_text_by_patent(
        csv_path=INPUT_CLAIMS_CSV,
        patent_id_columns=PATENT_ID_COLUMNS,
        text_columns=CLAIMS_TEXT_COLUMNS,
        delimiter=CSV_DELIMITER,
    ):
        if patent_id not in claims_by_patent:
            claims_by_patent[patent_id] = claims

    return claims_by_patent


def read_description_records():
    records = []

    for patent_id, description in iter_text_by_patent(
        csv_path=INPUT_DESCRIPTIONS_CSV,
        patent_id_columns=PATENT_ID_COLUMNS,
        text_columns=DESCRIPTION_TEXT_COLUMNS,
        delimiter=CSV_DELIMITER,
    ):
        if patent_id.upper().startswith('D'):
            continue
        if not description:
            continue

        records.append((patent_id, description))

    return records


def choose_patent(records, requested_patent_id):
    if not requested_patent_id:
        return random.choice(records)

    for patent_id, description in records:
        if patent_id == requested_patent_id:
            return patent_id, description

    raise ValueError('Patent not found in descriptions CSV: ' + requested_patent_id)


def print_section(title, text, max_chars):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)
    print(shorten_text(text, max_chars) if text else '[empty]')


def main():
    parser = argparse.ArgumentParser(
        description='Preview joined description + claims text for one patent.'
    )
    parser.add_argument(
        '--patent-id',
        help='Show a specific patent instead of choosing a random one.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Set a random seed so the random patent choice is repeatable.',
    )
    parser.add_argument(
        '--max-chars',
        type=int,
        default=3000,
        help='Maximum characters printed per text section. Use 0 for no limit.',
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    max_chars = None if args.max_chars == 0 else args.max_chars
    claims_by_patent = read_claims_lookup()
    description_records = read_description_records()

    if not description_records:
        raise RuntimeError('No description records found in ' + str(INPUT_DESCRIPTIONS_CSV))

    patent_id, description = choose_patent(description_records, args.patent_id)
    claims = claims_by_patent.get(patent_id, '')
    combined_text = '\n\n'.join(part for part in (description, claims) if part)

    print('Patent number: ' + patent_id)
    print('Descriptions CSV: ' + str(INPUT_DESCRIPTIONS_CSV))
    print('Claims CSV: ' + str(INPUT_CLAIMS_CSV))
    print('Description characters: ' + str(len(description)))
    print('Claims characters: ' + str(len(claims)))
    print('Combined characters: ' + str(len(combined_text)))

    print_section('DESCRIPTION / BODY', description, max_chars)
    print_section('CLAIMS', claims, max_chars)
    print_section('COMBINED TEXT: DESCRIPTION THEN CLAIMS', combined_text, max_chars)


if __name__ == '__main__':
    main()
