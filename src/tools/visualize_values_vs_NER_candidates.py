import json
import matplotlib.pyplot as plt

with open('data/spider/preprocessed_with_values_train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

    all_candidates = []
    for row in data:
        values = row['values']
        if values:
            candidates = row['ner_extracted_values_processed']
            all_candidates.append(candidates)
            print(f'Values ({len(values)}): {values}          Candidates ({len(candidates)}): {candidates}')

    print('Number of candidates sorted (search in this output to find which query it is):')
    print(sorted(map(lambda x: len(x), all_candidates), reverse=True))

    all_diff = []
    for row in data:
        if row['values']:
            diff = len(row['ner_extracted_values_processed']) - len(row['values'])

            # there are a few cases with < 0, which happens if the same value appears twice in a query.
            # No issue though, just leave it out to avoid confusion.
            if diff >= 0:
                all_diff.append(diff)

    plt.hist(all_diff, bins=range(0, 10))
    # plt.hist(all_diff, bins=range(1, 30))
    # plt.hist(all_diff, bins=range(10, 50))
    plt.show()
    print(f'Total difference: {sum(all_diff)} in {len(data)} queries.')

    # the following lines are just used t print out the values in the right format for the paper.
    n_diff = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    for diff in all_diff:
        n_diff[diff] += 1

    for key, value in n_diff.items():
        print(f'({key},{value})')
