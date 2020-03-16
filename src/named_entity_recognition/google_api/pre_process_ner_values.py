import argparse
import json
import os


def pre_process(entry):
    pre_processed_values = []
    for entity in entry['ner_extracted_values']['entities']:
        # for all types see https://cloud.google.com/natural-language/docs/reference/rest/v1beta2/Entity#Type
        # TODO: extend this pre-processing for e.g. ADDRESSES, PHONE_NUMBERS - see the link above.
        if entity['type'] == 'NUMBER':
            pre_processed_values.append(_compose_number(entity))
        if entity['type'] == 'DATE':
            pre_processed_values.append(_compose_date(entity))
        if entity['type'] == 'PRICE':
            pre_processed_values.append(_compose_price(entity))
        else:
            # just take the extracted value - without any adaptions
            pre_processed_values.append(entity['name'])

    # remove duplicates, which can appear due to the response from the google entities API
    return list(set(pre_processed_values))


def _compose_number(entity):
    # NUMBER will also detect e.g. a "one" and transform it to a 1 in the metadata
    return entity['metadata']['value']


def _compose_price(entity):
    # PRICE contains also the "currency" in the metadata. We assume that we don't need it.
    return entity['metadata']['value']


def _compose_date(entity):
    """
    This method formats returns a proper 'YYYY-MM-DD' string or a subpart of it (e.g. 'YYYY-MM') if not all information available.
    See Tests for more information.
    """
    full_date = ''
    if 'year' in entity['metadata']:
        full_date = entity['metadata']['year']

    if 'month' in entity['metadata']:
        if len(entity['metadata']['month']) == 1:
            month = '0' + entity['metadata']['month']
        else:
            month = entity['metadata']['month']

        if full_date:
            full_date = full_date + '-' + month
        else:
            full_date = month

    if 'day' in entity['metadata']:

        if len(entity['metadata']['day']) == 1:
            day = '0' + entity['metadata']['day']
        else:
            day = entity['metadata']['day']

        if full_date:
            full_date = full_date + '-' + day
        else:
            full_date = day

    return full_date


def are_all_values_found(expected, actual):
    all_values_found = True
    for value in expected:
        found = False
        for extracted_value in actual:
            if _is_value_equal(extracted_value, value):
                found = True
                break

        if not found:
            all_values_found = False
            print('Could not find {} in extracted values {}'.format(value, actual))

    return all_values_found, expected != []


def _is_value_equal(extracted_value, expected_value):
    # there are some cases were we have a float stored in the ground truth - even though we are actually looking for an int.
    if isinstance(expected_value, float) and expected_value.is_integer():
        expected_value = int(expected_value)

    expected_value = str(expected_value)
    return expected_value == extracted_value


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)

    args = arg_parser.parse_args()

    with open(os.path.join(args.data_path), 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    entry_with_values = 0
    not_found_count = 0
    for entry in data:
        extracted_values = pre_process(entry)
        entry['ner_extracted_values_processed'] = extracted_values
        all_values_found, has_values = are_all_values_found(entry['values'], entry['ner_extracted_values_processed'])

        if not all_values_found:
            not_found_count += 1
        if has_values:
            entry_with_values += 1

    print("Could find all values in {} of {} examples. {} entries contain values.".format(len(data) - not_found_count, len(data), entry_with_values))

    with open(os.path.join(args.output_path, 'ner_pre_processed_values.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f)