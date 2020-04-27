import re

from more_itertools import flatten

## ATTENTION ###
# the following heuristics are very basic and just a low effort approach a quite complex problem. The general idea is to
# add possible candidates for Values in questions which and not get delivered by a standard NER-system.
# This values will then be fed into the database-value-finder which will boil them down to the ones really found in the database.
# A more powerful idea would be to train a stochastic generative model with the task of generating value candidates based on some input.


def find_values_in_quote(question):
    """
    We try find all values in quotas, as this are almost always values we need later.
    We support a variety of different quota utf-8 characters.
    """
    matches = re.findall(r"\s[\"'‘“’](.+?)[\"'’”]", question)
    return [m for m in matches]


def find_ordinals(question_tokens):
    ordinals = {
        'once': 1,
        'twice': 2,
        'thrice': 3,

        'single': 1,
        'double': 2,
        'triple': 3,

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

    values_from_ordinals = []

    tokens = _sub_tokenize_tokens(question_tokens)

    for token in tokens:
        if token in ordinals:
            values_from_ordinals.append(str(ordinals[token]))

    return values_from_ordinals


def find_emails(question):
    matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", question)

    return [m for m in matches]


def find_genders(question_tokens):
    gender_abbrevations = {
        'female': ['F', 'female'],
        'females': ['F', 'female'],
        'girl': ['F', 'female'],
        'girls': ['F', 'female'],
        'male': ['M', 'male'],
        'males': ['M', 'male'],
        'boy': ['M', 'male'],
        'boys': ['M', 'male']
    }
    gender_values = []

    tokens = _sub_tokenize_tokens(question_tokens)

    for token in tokens:
        if token in gender_abbrevations:
            gender_values.extend(gender_abbrevations[token])

    return gender_values


def find_null_empty_values(question_tokens):
    null_empty_mentionings = {
        'NULL': 'null',
        'null': 'null',
        'empty': ''
    }

    null_empty_values = []

    tokens = _sub_tokenize_tokens(question_tokens)

    for token in tokens:
        if token in null_empty_mentionings:
            null_empty_values.append(null_empty_mentionings[token])

    return null_empty_values


def find_variety_of_common_mentionings(question_tokens):
    common_mentionings = {
        'spring': 'spring',
        'fall': 'fall',
        'summer': 'summer',
        'winter': 'winter',
        'morning': 'morning',
        'evening': 'evening',
        'night': 'night',
        'day': 'day',
        'yes': 'yes',
        'no': 'no'
    }

    common_values = []

    tokens = _sub_tokenize_tokens(question_tokens)

    for token in tokens:
        if token in common_mentionings:
            common_values.append(common_mentionings[token])

    return common_values


def find_special_codes(question):
    """
    Special codes refer to model numbers, classes, rooms, etc. Example:
    "What is the first name of the professor who is teaching CIS-220 and QM-261?"
    """
    matches = re.findall(r"[A-Z-/0-9]{2,}", question) # Attention: the space in the start of the regex ist not a mistake, but necessary to avoid apostrophes in word (e.g. Jean d'Arc is 'french')
    return [m for m in matches]


def find_single_letters(question):
    """
    if the word "letter" (and plural "letters") is mentioned, return all single letters
    """
    if re.findall(r"\bletter\b|\bletters\b", question):
        matches = re.findall(r"\b[A-Za-z]\b", question)

        return [m for m in matches]

    return []


def find_capitalized_words(question):
    """
    Often capitalized words are an indicator for a value. Capitalized words can also appear consecutive.
    Simple example: "What are the names and addressed of customers who have both New and Pending orders?" --> 'New' and 'Pending'
    Consecutive example: "What is id of the staff who had a Staff Department Assignment earlier than any Clerical Staff?" --> 'Staff Department Assignment' and 'Clerical Staff'
    """
    all_capitalized_words = []

    # with re.finditer() the group() property is referring to the full match. The next elements are the groups. Have a look at regex101.com to get the regex.
    consecutive_capitalized_words = [match.group() for match in re.finditer(r"(\b[A-Z0-9][A-Za-z0-9-/]+\b\s)+\b[A-Z0-9][A-Za-z0-9-/]+", question)]
    all_capitalized_words.extend(consecutive_capitalized_words)

    single_capitalized_word = [match.group() for match in re.finditer(r"\b[A-Z0-9][A-Za-z0-9-/]+\b", question)]

    for capitalized_word in single_capitalized_word:
        # avoid the first word of the sentence which is in english normally uppercase as well.
        if not question.startswith(capitalized_word):
            # make sure the capitalized word is not already part of consecutive_capitalized_words.
            if next(filter(lambda w: capitalized_word in w, consecutive_capitalized_words), None) is None:
                all_capitalized_words.append(capitalized_word)

    return all_capitalized_words


def find_months(question_tokens):
    months = {
        'january': '1/',
        'february': '2/',
        'march': '3/',
        'april': '4/',
        'may': '5/',
        'june': '6/',
        'july': '7/',
        'august': '8/',
        'september': '9/',
        'october': '10/',
        'november': '11/',
        'december': '12/'
    }

    months_fuzzy_dates = []

    tokens = _sub_tokenize_tokens(question_tokens)

    for token in tokens:
        if token in months:
            months_fuzzy_dates.append(months[token])

    return months_fuzzy_dates


def find_location_abbreviations(question):
    """
    This heuristic is just a very basic approximation for a much complexer problem. Location names are very divers and require a powerful
    model to understand them properly.
    """
    country_name_abbrevations_US = [
        'USA', 'US', 'United States', 'United States of America'
    ]

    country_name_abbrevations_UK = [
        'UK', 'United Kingdom', 'England'
    ]

    location_abbvreviations_US = {
        'AK': ['Alaska'],
        'AL': ['Alabama'],
        'AR': ['Arkansas'],
        'AZ': ['Arizona'],
        'CA': ['California'],
        'CO': ['Colorado'],
        'CT': ['Connecticut'],
        'DE': ['Delaware'],
        'FL': ['Florida'],
        'GA': ['Georgia'],
        'HI': ['Hawaii'],
        'IA': ['Iowa'],
        'ID': ['Idaho'],
        'IL': ['Illinois'],
        'IN': ['Indiana'],
        'KS': ['Kansas'],
        'KY': ['Kentucky'],
        'LA': ['Louisiana', 'Los Angeles'],
        'MA': ['Massachusetts'],
        'MD': ['Maryland'],
        'ME': ['Maine'],
        'MI': ['Michigan'],
        'MN': ['Minnesota'],
        'MO': ['Missouri'],
        'MS': ['Mississippi'],
        'MT': ['Montana'],
        'NC': ['North Carolin'],
        'ND': ['North Dakota'],
        'NE': ['Nebraska'],
        'NH': ['New Hampshire'],
        'NJ': ['New Jersey'],
        'NM': ['New Mexico'],
        'NV': ['Nevada'],
        'NY': ['New York'],
        'OH': ['Ohio'],
        'OK': ['Oklahoma'],
        'OR': ['Oregon'],
        'PA': ['Pennsylvania'],
        'RI': ['Rhode Island'],
        'SC': ['South Carolin'],
        'SD': ['South Dakota'],
        'TN': ['Tennessee'],
        'TX': ['Texas'],
        'UT': ['Utah'],
        'VA': ['Virginia'],
        'VT': ['Vermont'],
        'WA': ['Washington'],
        'WI': ['Wisconsin'],
        'WV': ['West Virginia'],
        'WY': ['Wyoming']
    }

    location_candidates = []

    for key, potential_values in location_abbvreviations_US.items():
        add_me = False
        if key in question:
            add_me = True

        for sub_value in potential_values:
            if sub_value in question:
                add_me = True

        if add_me:
            location_candidates.append(key)
            location_candidates.extend(potential_values)

    for abbreviation in country_name_abbrevations_US:
        if abbreviation in question:
            # we don't know how to look for USA - therefore add all options. The database finder should sort them out.
            location_candidates.extend(country_name_abbrevations_US)

    for abbreviation in country_name_abbrevations_UK:
        if abbreviation in question:
            # we don't know how to look for United Kingdom - therefore add all options. The database finder should sort them out.
            location_candidates.extend(country_name_abbrevations_UK)

    return location_candidates


def _sub_tokenize_tokens(tokens):
    """
    There are some combined tokens we need to further tokenize (example: "fourth-grade")
    """
    return flatten(map(lambda t: t.split('-'), tokens))
