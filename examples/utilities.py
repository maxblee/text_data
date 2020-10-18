"""A small module designed for loading in data for the example notebooks."""
import functools
import glob
import re

from text_data import tokenize


def load_sotu_data():
    """This loads data from State of the Union Addresses."""
    sotu_files = glob.glob("sotu-data/*.txt")
    path_desc = re.compile(r"sotu-data/([A-Za-z]+)_([0-9]{4})\.txt")
    for filepath in sotu_files:
        with open(filepath, "r") as f:
            raw_text = f.read()
        pres, year = path_desc.search(filepath).groups()
        yield {"president": pres, "year": year, "speech": raw_text}


def tokenizer(document: str) -> tokenize.TokenizeResult:
    """Simple tokenizer for all of the State of the Union Addresses."""
    word_tokenizer = functools.partial(
        tokenize.tokenize_regex_positions, r"(?:^|\b)[A-Za-z]+(?:$|\b)"
    )
    return tokenize.postprocess_positions([str.lower], word_tokenizer(document))
