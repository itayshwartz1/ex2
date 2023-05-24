from __future__ import print_function
import collections
import re
import random


##############################################################################
# Program parameters

# Path to the text file containing the ciphertext
INFILE = 'enc.txt'

# Path to the text file containing the reference text
REFFILE = 'paradiso.txt'

# Encrypted chars in the ciphertext
CHARS = 'ABCDEFGHILMNOPQRSTUVZ'

# Size of the population to use for the genetic algorithm
POPULATION_SIZE = 50

# Size of the population slice of best peforming solutions to keep at each
# iteration
TOP_POPULATION = 10

# Number of intervals for which the best score has to be stable before aborting
# the genetic algorith
STABILITY_INTERVALS = 20

# Number of crossovers to execute for each new child in the genetic algorithm
CROSSOVER_COUNT = 2

# Number of random mutation to introduce for each new child in the genetic
# algorithm
MUTATIONS_COUNT = 1


##############################################################################
# Implementation

def pairwise(iterable):
    prev = None

    for item in iterable:
        if prev is not None:
            yield prev + item
        prev = item


def bigram(text):
    counter = collections.Counter()
    words = re.sub('[^{}]'.format(CHARS), ' ', text).split()

    for word in words:
        for pair in pairwise(word):
            counter[pair] += 1

    return counter


def decode(ciphertext, key):
    cleartext = ''
    for char in ciphertext:
        cleartext += key.get(char, char)
    return cleartext


def init_mapping():
    # Generate a randomly initialized solution
    repls = set(CHARS)
    mapping = {}
    for c in CHARS:
        if c in mapping:
            continue
        repl = random.choice(list(repls))
        repls.remove(repl)
        repls.discard(c)
        mapping[c] = repl
        mapping[repl] = c
    return mapping


def update_mapping(mapping, char, repl):
    # Update the solution by switching `char` with `repl`
    # and `repl` with `char`.
    current_repl = mapping[char]
    current_char = mapping[repl]

    if current_char == repl:
        current_char = current_repl
    elif current_repl == char:
        current_repl = current_char

    mapping[current_char] = current_repl
    mapping[current_repl] = current_char

    mapping[char] = repl
    mapping[repl] = char


##############################################################################
# Genetic algorithm routines

def select(population, ciphertext, ref_bigram):
    scores = []

    # Compute the score of each solution
    for p in population:
        scores.append((score(decode(ciphertext, p), ref_bigram), p))

    # Sort the solutions by their score
    sorted_population = sorted(scores, reverse=True)

    # Select only the best TOP_POPULATION solutions
    selected_population = sorted_population[:TOP_POPULATION]

    return selected_population[0][0], [m for _, m in selected_population]


def generate(population):
    new_population = population[:]
    while len(new_population) < POPULATION_SIZE:
        # Randomly select two parent solutions
        x, y = random.choice(population), random.choice(population)

        # Create the child solution
        child = x.copy()

        # Switch CROSSOVER_COUNT chromosomes between the parents
        for i in range(CROSSOVER_COUNT):
            char = random.choice(list(CHARS))
            update_mapping(child, char, y[char])

        # Randomly mutate MUTATIONS_COUNT chromosomes of the the child solution
        for i in range(MUTATIONS_COUNT):
            char = random.choice(list(CHARS))
            repl = random.choice(list(CHARS))
            update_mapping(child, char, repl)

        # Add the newly obtained child the the current population
        new_population.append(child)
    return new_population


def score(text, ref_bigram):
    text_bigram = bigram(text)
    score = 0

    # Multiply the number of occurrences of each pair in the decoded
    # ciphertext with the number of occurrences of that same pair in the
    # reference text, then take the sum of all multiplications.
    for pair, occurrences in text_bigram.items():
        score += occurrences * ref_bigram[pair]

    return score


###############################################################################
# Decryption routine

def decrypt():
    # Read the reference text into memory
    with open(REFFILE) as fh:
        reftext = fh.read().upper()

    # Analyze the reference text and compute a mapping of each pair of letters
    # to the number of occurrences in the reference text
    #
    #    ref_bigram = {
    #        'AA': 1,
    #        'AB': 64,
    #        'AC': 354,
    #        'AD': 279,
    #        'AE': 26,
    #        'AF': 52,
    #        'AG': 241,
    #        'AH': 2,
    #        'AI': 260,
    #        'AL': 1141,
    #        'AM': 353,
    #        ...
    #        'VE': 958,
    #        'VI': 727,
    #        'VO': 409,
    #        'VR': 43,
    #        'VU': 33,
    #        'VV': 28,
    #        'ZA': 249,
    #        'ZE': 35,
    #        'ZI': 240,
    #        'ZO': 49,
    #        'ZZ': 103,
    #    }
    #
    ref_bigram = bigram(reftext)

    # Read the ciphertext into memory
    with open(INFILE) as fh:
        ciphertext = fh.read().upper()

    # Create an initial population of random possible solutions
    population = [init_mapping() for i in range(POPULATION_SIZE)]
    print(population[0])

    # Set the initial values for the stability checker
    last_score = 0
    last_score_increase = 0
    iterations = -STABILITY_INTERVALS

    # Run the genetic algorithm
    while last_score_increase < STABILITY_INTERVALS:
        # Fill up the population up to POPULATION_SIZE solutions by crossing
        # over and mutating the TOP_POPULATION best solutions
        population = generate(population)

        # Select the TOP_POPULATION best solutions from the current population
        best_score, population = select(population, ciphertext, ref_bigram)

        # Update the stability check state with the current best score
        if best_score > last_score:
            last_score_increase = 0
            last_score = best_score
        else:
            last_score_increase += 1
        print(best_score)

        iterations += 1

    # Print the current (best) solution
    #
    #     best_solution = population[0] = {
    #         'A': 'M',
    #         'B': 'O',
    #         'C': 'Q',
    #         'D': 'R',
    #         'E': 'P',
    #         'F': 'T',
    #         'G': 'G',  # Wrong, should be 'Z'
    #         'H': 'U',
    #         'I': 'V',
    #         'L': 'S',
    #         'M': 'A',
    #         'N': 'N',
    #         'O': 'B',
    #         'P': 'E',
    #         'Q': 'C',
    #         'R': 'D',
    #         'S': 'L',
    #         'T': 'F',
    #         'U': 'H',
    #         'V': 'I',
    #         'Z': 'Z',  # Wrong, should be 'G'
    #     }
    #
    print('Best solution found after {} iterations:'.format(iterations))
    print(decode(ciphertext, population[0]))
    print(population[0])


decrypt()