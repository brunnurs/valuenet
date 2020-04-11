from dataclasses import dataclass


@dataclass(init=True)
class NerExtractionData:
    heuristic_values_in_quota: list
    heuristic_ordinals: list
    heuristics_emails: list

    ner_dates: list
    ner_numbers: list
    ner_prices: list
    ner_remaining: list
