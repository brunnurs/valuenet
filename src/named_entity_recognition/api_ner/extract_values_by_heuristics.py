import re

ORDINALS_AND_SIMILAR = {
    'once': 1,
    'twice': 2,
    'thrice': 3,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 9,
}


def find_values_in_quota(question):
    """
    We try find all values in quotas, as this are almost always values we need later.
    We support a variaty of different quota ascii characters.
    """
    matches = re.findall(r" [\"'‘“](.+?)[\"'’”]", question) # Attention: the space in the start of the regex ist not a mistake, but necessary to avoid apostrophes in word (e.g. Jean d'Arc is 'french')
    return [m for m in matches] + [f'%{m}%' for m in matches]


def find_ordinals(question_tokens):
    values_from_ordinals = []

    for token in question_tokens:
        if token in ORDINALS_AND_SIMILAR:
            values_from_ordinals.append(str(ORDINALS_AND_SIMILAR[token]))

    return values_from_ordinals


def find_emails(question):
    matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", question)

    return [m for m in matches]
