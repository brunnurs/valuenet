import json
import matplotlib.pyplot as plt

with open('data/spider/preprocessed_with_values_dev.json', 'r', encoding='utf-8') as json_file:
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

    plt.hist(all_diff, bins=range(0, 30))
    # plt.hist(all_diff, bins=range(1, 30))
    # plt.hist(all_diff, bins=range(10, 50))
    plt.show()
    print(f'Total difference: {sum(all_diff)} in {len(data)} queries.')