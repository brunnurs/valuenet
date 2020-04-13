from unittest import TestCase

from named_entity_recognition.handcrafted_heuristics import find_values_in_quote, find_ordinals, \
    find_emails, find_genders, find_null_empty_values, find_variety_of_common_mentionings, find_special_codes, \
    find_single_letters, find_capitalized_words, find_months, find_location_abbreviations


class Test(TestCase):
    def test__find_values_in_quota(self):
        # GIVEN
        question = "Find the names of the customers who have order status both 'On Road' and \"Shipped\""
        # WHEN
        values = find_values_in_quote(question)

        # THEN
        self.assertEqual(['On Road',
                          'Shipped'], values)

    def test__find_values_in_quota_apostroph_in_names(self):
        # GIVEN
        question = "Which head's name has the substring 'Ha'? List the id and name."
        # WHEN
        values = find_values_in_quote(question)

        # THEN
        self.assertEqual(['Ha'], values)

    def test__find_values_in_quota__another_weird_apostroph(self):
        # GIVEN
        question = "display the employee number and name( first name and last name ) for all employees who work in a department with any employee whose name contains a ’T’."
        # WHEN
        values = find_values_in_quote(question)

        # THEN
        self.assertEqual(['T'], values)

    def test__find_ordinals(self):
        # GIVEN
        question = ['how', 'many', 'third', 'head', 'of', 'department', 'are', 'older', 'than', '56', '?']

        # WHEN
        ordinals = find_ordinals(question)

        # THEN
        self.assertEqual(['3'], ordinals)

    def test__find_ordinals_combined_tokens(self):
        # GIVEN
        question = ["Report",
                    "the",
                    "total",
                    "number",
                    "of",
                    "students",
                    "for",
                    "each",
                    "fourth-grade",
                    "classroom",
                    "."]

        # WHEN
        ordinals = find_ordinals(question)

        # THEN
        self.assertEqual(['4'], ordinals)

    def test__find_emails(self):
        # GIVEN
        question = 'Find id of the candidate whose email is stanley.monahan@example.org?'

        # WHEN
        ordinals = find_emails(question)

        # THEN
        self.assertEqual(['stanley.monahan@example.org'], ordinals)

    def test__find_genders(self):
        # GIVEN
        question = ["how",
                    "many",
                    "female",
                    "people",
                    "are",
                    "older",
                    "than",
                    "30",
                    "in",
                    "our",
                    "record",
                    "?"]

        # WHEN
        gender_values = find_genders(question)

        # THEN
        self.assertEqual(['F', 'female'], gender_values)

    def test__find_genders_plural(self):
        # GIVEN
        question = [
            "show",
            "name",
            "for",
            "all",
            "female",
            "from",
            "canada",
            "having",
            "a",
            "wedding",
            "in",
            "year",
            "year",
            "."
        ]
        # WHEN
        gender_values = find_genders(question)

        # THEN
        self.assertEqual(['F', 'female'], gender_values)

    def test__find_genders_male_plural(self):
        # GIVEN
        question = [
            "find",
            "average",
            "height",
            "and",
            "weight",
            "for",
            "all",
            "male",
            "(",
            "sex",
            "is",
            "m",
            ")",
            "."
        ]

        # WHEN
        gender_values = find_genders(question)

        # THEN
        self.assertEqual(['M', 'male'], gender_values)

    def test__find_genders_girl(self):
        # GIVEN
        question = [
            "how",
            "many",
            "girl",
            "student",
            "who",
            "are",
            "younger",
            "than",
            "25",
            "?"
        ]

        # WHEN
        gender_values = find_genders(question)

        # THEN
        self.assertEqual(['F', 'female'], gender_values)

    def test__find_null_empty_values(self):
        # GIVEN
        question = [
            "find",
            "name",
            "of",
            "all",
            "reviewer",
            "who",
            "have",
            "rating",
            "with",
            "a",
            "null",
            "value",
            "for",
            "date",
            "."
        ]
        # WHEN
        null_empty_values = find_null_empty_values(question)

        # THEN
        self.assertEqual(['null'], null_empty_values)

    def test__find_null_empty_values_empty_value(self):
        # GIVEN
        question = [
            "what",
            "are",
            "first",
            "name",
            "and",
            "last",
            "name",
            "of",
            "player",
            "whose",
            "death",
            "record",
            "is",
            "empty",
            "?"
        ]

        # WHEN
        null_empty_values = find_null_empty_values(question)

        # THEN
        self.assertEqual([''], null_empty_values)

    def test__find_variaty_of_common_mentionings(self):
        # GIVEN
        # in opposite to the other tests not a real sample - just to make sure we get all values.
        question = [
            "what",
            "are",
            "first",
            "spring",
            "and",
            "fall",
            "summer",
            "winter",
            "player",
            "whose",
            "morning",
            "evening",
            "is",
            "night",
            "day",
            "?",
            "yes",
            "no",
        ]

        # WHEN
        common_mentionings = find_variety_of_common_mentionings(question)

        # THEN
        self.assertEqual(['spring', 'fall', 'summer', 'winter', 'morning', 'evening', 'night', 'day', 'yes', 'no'], common_mentionings)

    def test__find_special_codes(self):
        # GIVEN
        question1 = "What is the first name of the professor who is teaching CIS-220 and QM-261?"
        question2 = "What are the codes of all the courses that are located in room KLR209?"
        question3 = "How many departments are in the division AS?"

        # WHEN
        special_codes1 = find_special_codes(question1)
        special_codes2 = find_special_codes(question2)
        special_codes3 = find_special_codes(question3)

        # THEN
        self.assertEqual(['CIS-220', 'QM-261'], special_codes1)
        self.assertEqual(['KLR209'], special_codes2)
        self.assertEqual(['AS'], special_codes3)

    def test__find_single_letters(self):
        # GIVEN
        question1 = "when is the hire date for those employees whose first name does not containing the letter M?"
        question2 = "Display all the information for all employees who have the letters D or S in their first name and also arrange the result in descending order by salary."
        question3 = "What are the descriptions of the categories that products with product descriptions that contain the letter t are in?"
        question4 = "A sentence without the special words has no matches."

        # WHEN
        special_codes1 = find_single_letters(question1)
        special_codes2 = find_single_letters(question2)
        special_codes3 = find_single_letters(question3)
        special_codes4 = find_single_letters(question4)

        # THEN
        self.assertEqual(['M'], special_codes1)
        self.assertEqual(['D', 'S'], special_codes2)
        self.assertEqual(['t'], special_codes3)
        self.assertEqual([], special_codes4)

    def test__find_capitalized_words(self):
        # GIVEN
        question1 = "For grants that have descriptions of Regular and Initial Applications, what are their start dates?"
        question2 = "Find the department name and room of the course INTRODUCTION TO COMPUTER SCIENCE."
        question3 = "What are the names and addressed of customers who have both New and Pending orders?"
        question4 = "Show the shipping charge and customer id for customer orders with order status Cancelled or Paid."
        question5 = "Find the names of departments that are either in division AS or in division EN and in Building NEB."
        question6 = "Return the distinct name of customers whose order status is Pending, in the order of customer id."
        question7 = "What is id of the staff who had a Staff Department Assignment earlier than any Clerical Staff?"
        question8 = "List the order id, customer id for orders in Cancelled status, ordered by their order dates."
        question9 = "What is the student capacity and type of gender for the dorm whose name as the phrase Donor in it?"
        question10 = "Do you have a friend living in New York City?"
        question11 = "Show all flight numbers with aircraft Airbus A340-300."
        question12 = "Show names for all employees who have certificates on both Boeing 737-800 and Airbus A340-300"

        # WHEN
        capitalized_words1 = find_capitalized_words(question1)
        capitalized_words2 = find_capitalized_words(question2)
        capitalized_words3 = find_capitalized_words(question3)
        capitalized_words4 = find_capitalized_words(question4)
        capitalized_words5 = find_capitalized_words(question5)
        capitalized_words6 = find_capitalized_words(question6)
        capitalized_words7 = find_capitalized_words(question7)
        capitalized_words8 = find_capitalized_words(question8)
        capitalized_words9 = find_capitalized_words(question9)
        capitalized_words10 = find_capitalized_words(question10)
        capitalized_words11 = find_capitalized_words(question11)
        capitalized_words12 = find_capitalized_words(question12)

        # THEN
        self.assertEqual(['Initial Applications', 'Regular'], capitalized_words1)
        self.assertEqual(['INTRODUCTION TO COMPUTER SCIENCE'], capitalized_words2)
        self.assertEqual(['New', 'Pending'], capitalized_words3)
        self.assertEqual(['Cancelled', 'Paid'], capitalized_words4)
        self.assertEqual(['Building NEB', 'AS', 'EN'], capitalized_words5)
        self.assertEqual(['Pending'], capitalized_words6)
        self.assertEqual(['Staff Department Assignment', 'Clerical Staff'], capitalized_words7)
        self.assertEqual(['Cancelled'], capitalized_words8)
        self.assertEqual(['Donor'], capitalized_words9)
        self.assertEqual(['New York City'], capitalized_words10)
        self.assertEqual(['Airbus A340-300'], capitalized_words11)
        self.assertEqual(['Boeing 737-800', 'Airbus A340-300'], capitalized_words12)

    def test__find_location_abbreviations(self):
        # GIVEN
        question1 = "What are the departure and arrival dates of all flights from LA to Honolulu?"
        question2 = "List the number of invoices from the US, grouped by state."
        question3 = "List the number of invoices and the invoice total from California."
        question4 = "What are the companies and main industries of all companies that are not headquartered in the United States?"
        question5 = "What are the names of the ships  that are from either the US or the UK?"
        question6 = "What is average number of students enrolled in Florida colleges?"
        question7 = "What are the names and enrollment numbers for colleges that have more than 10000 enrolled and are located in Louisiana?"

        # WHEN
        location_abbreviations1 = find_location_abbreviations(question1)
        location_abbreviations2 = find_location_abbreviations(question2)
        location_abbreviations3 = find_location_abbreviations(question3)
        location_abbreviations4 = find_location_abbreviations(question4)
        location_abbreviations5 = find_location_abbreviations(question5)
        location_abbreviations6 = find_location_abbreviations(question6)
        location_abbreviations7 = find_location_abbreviations(question7)

        # THEN
        self.assertEqual(['LA', 'Louisiana', 'Los Angeles'], location_abbreviations1)
        self.assertEqual(['USA', 'US', 'United States', 'United States of America'], location_abbreviations2)
        self.assertEqual(['CA', 'California'], location_abbreviations3)
        self.assertEqual(['USA', 'US', 'United States', 'United States of America'], location_abbreviations4)
        self.assertEqual(['USA', 'US', 'United States', 'United States of America', 'UK', 'United Kingdom', 'England'], location_abbreviations5)
        self.assertEqual(['FL', 'Florida'], location_abbreviations6)
        self.assertEqual(['LA', 'Louisiana', 'Los Angeles'], location_abbreviations7)

    def test__find_months(self):
        # GIVEN
        question = [
            "for",
            "each",
            "zip",
            "code",
            ",",
            "return",
            "average",
            "mean",
            "temperature",
            "of",
            "august",
            "there",
            "."
        ]
        # WHEN
        months_fuzzy = find_months(question)

        # THEN
        self.assertEqual(['8/'], months_fuzzy)
