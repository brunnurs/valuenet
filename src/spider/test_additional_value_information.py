from unittest import TestCase


class TestAdditionalValueInformation(TestCase):
    def test_enrich_value_by_additional_information(self):
        # GIVEN
        additional_value_information = [
            [
                "2014",
                "Year",
                "concert"
            ],
            [
                "2014",
                "Song_release_year",
                "singer"
            ]
        ]

        # WHEN

        # THEM
