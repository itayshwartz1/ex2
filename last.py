import random
import string
import re
from collections import Counter

# Assuming the file is in the same directory
encrypted_file = "enc.txt"
dictionary_file = "dict.txt"
plain_text_file = "plain.txt"
perm_file = "perm.txt"
letter_freq_file = "Letter_Freq.txt"
letter2_freq_file = "Letter2_Freq.txt"

# Load dictionary
with open(dictionary_file, 'r') as file:
    dictionary = set(file.read().splitlines())

# Load encrypted text
with open(encrypted_file, 'r') as file:
    encrypted_text = file.read().strip()
    num_of_words = len(encrypted_text)

frequencies = {}
with open(letter_freq_file, 'r') as file:
    for line in file:
        probability, letter = line.split()
        frequencies[letter.lower()] = float(probability)

frequencies2 = {}
with open(letter2_freq_file, 'r') as file:
    for line in file :
        line = line.strip()
        if line == "":
            break
        probability, letter_pair = line.split()
        frequencies2[letter_pair.lower()] = float(probability)


# Genetic algorithm parameters
POPULATION_SIZE = 500
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.3

# Generate initial population
population = [''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))
              for _ in range(POPULATION_SIZE)]





# Fitness function
def fitness(individual):
    decrypted_text = decrypt(encrypted_text, individual)
    words = re.findall(r'\b\w+\b', decrypted_text)
    valid_words = sum(word in dictionary for word in words)
    hit_rate = valid_words / num_of_words

    # initial dict of pairs of letters, all the value are 0
    pair_dict = {}
    letter_dict = {}
    for letter1 in string.ascii_lowercase:
        letter_dict[letter1] = 0
        for letter2 in string.ascii_lowercase:
            pair = letter1 + letter2
            pair_dict[pair] = 0

    for word in words:
        word = word.strip()
        for letter in word:
            letter_dict[letter] += 1
        for i in range(len(word) - 1):
            pair = word[i: i + 2]
            pair_dict[pair] += 1

    sum_of_letters = sum(letter_dict.values())
    sum_letter_freq = 0
    for letter in letter_dict:
        val = letter_dict[letter] / sum_of_letters
        sum_letter_freq += abs(frequencies[letter] - val)

    sum_of_pairs = sum(pair_dict.values())
    sum_pair_freq = 0
    for pair in pair_dict:
        val = pair_dict[pair] / sum_of_pairs
        sum_pair_freq += abs(frequencies2[pair] - val)

    return hit_rate - sum_letter_freq * 10 - sum_pair_freq * 10 + 100


# Fitness function
def fitness2(individual):
    decrypted_text = decrypt(encrypted_text, individual)
    words = re.findall(r'\b\w+\b', decrypted_text)
    valid_words = sum(word in dictionary for word in words)
    hit_rate = valid_words / num_of_words

    return hit_rate


# Decrypt function
def decrypt(text, key):
    table = str.maketrans(string.ascii_lowercase, key)
    return text.translate(table)

# Selection function
def selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=fitness, reverse=True)
    return tournament[0]

# Crossover function
def crossover(parent1, parent2):
    index = random.randint(0, 25)

    # Convert parent1 and parent2 to lists
    child1 = list(parent1)
    child2 = list(parent2)

    # Create sets to track used and unused letters
    letters_set1 = set(string.ascii_lowercase)
    letters_set2 = set(string.ascii_lowercase)

    # Remove mapped letters from sets
    for j in range(index):
        letters_set1.remove(parent1[j])
        letters_set2.remove(parent2[j])

    # Perform crossover
    for i in range(index, 26):
        if parent2[i] not in letters_set1:
            child1[i] = '$'
        else:
            child1[i] = parent2[i]
            letters_set1.remove(parent2[i])

        if parent1[i] not in letters_set2:
            child2[i] = '$'
        else:
            child2[i] = parent1[i]
            letters_set2.remove(parent1[i])

    # Fill in the remaining positions with unused letters
    for t in range(0, 26):
        if child1[t] == '$':
            child1[t] = letters_set1.pop()

        if child2[t] == '$':
            child2[t] = letters_set2.pop()

    # Convert child1 and child2 back to strings
    child1 = ''.join(child1)
    child2 = ''.join(child2)

    return child1, child2

# Mutation function
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index1, index2 = random.sample(range(len(individual)), 2)
        individual = list(individual)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return ''.join(individual)

# Main genetic algorithm loop
steps = 0
stack = [0] * 10
while True:
    steps += 1

    population.sort(key=fitness, reverse=True)

    # stack[steps % 10] = population[0]
    # big_muted = False
    # if sum(population[0] == value for value in stack) == 7:
    #     big_muted = True
    # if all(population[0] == value for value in stack):

    hit_rate = fitness2(population[0])
    print(hit_rate)
    next_population = population[:round(POPULATION_SIZE * 0.1)]  # elitism

    while len(next_population) < POPULATION_SIZE:
        parent1 = selection(population)
        parent2 = selection(population)
        child1, child2 = crossover(parent1, parent2)
        next_population += [mutate(child1), mutate(child2)]

    population = next_population

# Write the decrypted text and permutation to the output files
decryption_key = population[0]

with open(plain_text_file, 'w') as file:
    file.write(decrypt(encrypted_text, decryption_key))

with open(perm_file, 'w') as file:
    for original, new in zip(string.ascii_lowercase, decryption_key):
        file.write(f"{original} -> {new}\n")

print(f"Decryption completed in {steps} steps")