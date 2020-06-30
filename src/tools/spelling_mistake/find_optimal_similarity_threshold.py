import psycopg2
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pytictoc import TicToc


def load_spelling_mistake_dataset():
    misspelled_words = {}
    with open("src/tools/spelling_mistake/missp.dat", encoding="utf-8") as file:
        current_word = ''
        missspellings = []

        for word in file:
            if word.startswith('$'):
                if missspellings:
                    misspelled_words[current_word] = missspellings
                missspellings = []
                current_word = word[1:-1]
            else:
                missspellings.append(word[:-1])

    return misspelled_words


def calculate_similarities():
    similarities = []

    conn = psycopg2.connect(
        database="cordis",
        user="postgres",
        host="localhost",
        password="postgres"
    )
    # Open a cursor to perform database operations
    cur = conn.cursor()

    t = TicToc()
    t.tic()
    for word, misspellings in misspelled_words.items():

        word = word.replace("'", "")

        for misspelling in misspellings:
            misspelling = misspelling.replace("'", "")

            # Query the database and obtain data as Python objects
            cur.execute(f"SELECT similarity('{word}','{misspelling}')")
            # cur.execute(f"SELECT word_similarity('{word}','{misspelling}')")
            # cur.execute(f"SELECT levenshtein('{word}','{misspelling}')")
            # cur.execute(f"SELECT difference('{word}','{misspelling}')")         # difference is using the soundex algorithm
            result = cur.fetchone()
            print(f'{word}  {misspelling}   {result[0]}')
            similarities.append(result[0])

            # Close communication with the database
    cur.close()
    conn.close()
    t.toc(f"Calculating all {len(similarities)} similarities took")
    return similarities


if __name__ == '__main__':
    misspelled_words = load_spelling_mistake_dataset()

    similarities = calculate_similarities()
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.hist(similarities, bins=100)
    loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    plt.show()

