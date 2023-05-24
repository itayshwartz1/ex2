import random
import string
from collections import Counter

# Assuming the file is in the same directory
encrypted_file = "enc.txt"
dictionary_file = "dict.txt"
plain_text_file = "plain.txt"
perm_file = "perm.txt"

# Load dictionary
with open(dictionary_file, 'r') as file:
    dictionary = set(file.read().splitlines())

# Load encrypted text
with open(encrypted_file, 'r') as file:
    encrypted_text = file.read()

# Genetic algorithm parameters
POPULATION_SIZE = 1000
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.2
BIG_MUTATION_RATE = 0.9
ELITISM_RATE = 0.1
LETTER_RATE = 2 / 26
BIG_LETTER_RATE = 10 / 26

# Generate initial population
population = [''.join(random.sample(string.ascii_lowercase, len(string.ascii_lowercase)))
              for _ in range(POPULATION_SIZE)]

# Fitness function
def fitness(individual):
    decrypted_text = decrypt(encrypted_text, individual)
    words = decrypted_text.split()
    valid_words = sum(word in dictionary for word in words)
    return valid_words / len(words)

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
    cut = random.randint(0, len(parent1))
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return child1, child2

def mutate(individual, big_muted):
    muted_rate = MUTATION_RATE
    leter_rate = LETTER_RATE
    letters = list(string.ascii_lowercase)
    if is_big_muted:
        muted_rate = BIG_MUTATION_RATE
        leter_rate = BIG_LETTER_RATE
    if random.random() < muted_rate:
        for i in range(26):
            if random.random() < leter_rate:
                index1, index2 = random.sample(range(len(individual)), 2)
                individual = list(individual)
                individual[index1], individual[index2] = individual[index2], individual[index1]
    return ''.join(individual)


# Mutation function
# def mutate(individual, is_big_muted):
#     muted_rate = MUTATION_RATE
#     if is_big_muted:
#         muted_rate = BIG_MUTATION_RATE
#     if random.random() < muted_rate:
#         index1, index2 = random.sample(range(len(individual)), 2)
#         individual = list(individual)
#         individual[index1], individual[index2] = individual[index2], individual[index1]
#     return ''.join(individual)

# Main genetic algorithm loop
steps = 0
stack = [0] * 10
while steps < 1000:
    steps += 1
    is_big_muted = False
    population.sort(key=fitness, reverse=True)
    best = fitness(population[0])
    stack[steps % 10] = best
    if all(best == value for value in stack):
        if best < 0.8:
            is_big_muted = True
        else:
            break

    print(str(best))
    if best == 1:
        break

    tmp_population = population[:round(POPULATION_SIZE * ELITISM_RATE)]
    next_population = []
    for privilege in next_population:
        next_population.append(mutate(privilege, is_big_muted))

    while len(next_population) < POPULATION_SIZE:
        parent1 = selection(population)
        parent2 = selection(population)
        child1, child2 = crossover(parent1, parent2)
        next_population += [mutate(child1, is_big_muted), mutate(child2, is_big_muted)]

    population = next_population

# Write the decrypted text and permutation to the output files
decryption_key = population[0]

with open(plain_text_file, 'w') as file:
    file.write(decrypt(encrypted_text, decryption_key))

with open(perm_file, 'w') as file:
    for original, new in zip(string.ascii_lowercase, decryption_key):
        file.write(f"{original} -> {new}\n")

print(f"Decryption completed in {steps} steps")