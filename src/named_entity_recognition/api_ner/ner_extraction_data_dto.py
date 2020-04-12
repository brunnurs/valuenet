from dataclasses import dataclass


@dataclass(init=True)
class NerExtractionData:
    heuristic_values_in_quote: list
    heuristic_ordinals: list
    heuristics_emails: list
    heuristics_genders: list
    heuristics_null_empty: list
    heuristics_variety_common_mentionings: list
    heuristics_special_codes: list
    heuristics_single_letters: list
    heuristics_capitalized_words: list
    heuristics_months: list
    heuristics_location_abbreviations: list

    ner_dates: list
    ner_numbers: list
    ner_prices: list
    ner_remaining: list
