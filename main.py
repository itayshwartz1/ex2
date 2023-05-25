'''
Itay Shwartz 318528171
Noa Eitan 316222777
'''
import random
import string
import re
import time
import csv
import matplotlib.pyplot as plt

NUM_BEST_TO_DUPLICATE = 0.2
MUTATE_RATE = 0.2
BIG_MUTATE_RATE = 0.7
LETTER_TO_CHANGE = 2 / 26
BIG_LETTER_TO_CHANGE = 8 / 26
MAX_GENERATION = 1000
POPULATION_SIZE = 500
global LEN_ENC
LEN_ENC = 0
global FITNESS_COUNT
FITNESS_COUNT = 0

global AVG
AVG = []

global BEST
BEST = []

global WORST
WORST = []


def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip().lower() for line in file)


def load_letter_frequencies(file_path):
    frequencies = {}
    with open(file_path, 'r') as file:
        for line in file:
            probability, letter = line.split()
            frequencies[letter.lower()] = float(probability)
    return frequencies


def load_letter_pair_frequencies(file_path):
    frequencies = {}
    with open(file_path, 'r') as file:
        for line in file :
            line = line.strip()
            if line == "":
                break
            probability, letter_pair = line.split()
            frequencies[letter_pair.lower()] = float(probability)

    return frequencies


def check_duplicate_values(mapping):
    value_count = {}
    for value in mapping.values():
        if value in value_count:
            return True  # Duplicate value found
        value_count[value] = 1
    return False


def generate_random_mapping():
    letters = list(string.ascii_lowercase)
    random.shuffle(letters)
    sol = dict(zip(string.ascii_lowercase, letters))
    return sol


def decrypt_text(text, mapping):
    decrypted_text = ''
    for char in text:
        decrypted_text += mapping.get(char, char)
    return decrypted_text


def calculate_fitness(decrypted_text, dictionary, letter_frequencies, letter_pair_frequencies):
    global FITNESS_COUNT
    FITNESS_COUNT += 1
    words = re.findall(r'\b\w+\b', decrypted_text)
    ascii_dict = {ascii_value: 0 for ascii_value in string.ascii_lowercase}

    # initial dict of pairs of letters, all the value are 0
    pair_dict = {}
    for letter1 in string.ascii_lowercase:
        for letter2 in string.ascii_lowercase:
            pair = letter1 + letter2
            pair_dict[pair] = 0

    num_of_hits = 0
    for word in words:
        word = word.strip()
        if word in dictionary:
            num_of_hits += 1
        for letter in word:
            ascii_dict[letter] += 1
        for i in range(len(word) - 1):
            pair = word[i: i + 2]
            pair_dict[pair] += 1

    sum_of_letters = sum(ascii_dict.values())
    sum_letter_freq = 0
    for letter in ascii_dict:
        val = ascii_dict[letter] / sum_of_letters
        sum_letter_freq += abs(letter_frequencies[letter] - val) ** 2

    sum_of_pairs = sum(pair_dict.values())
    sum_pair_freq = 0
    for pair in pair_dict:
        val = pair_dict[pair] / sum_of_pairs
        sum_pair_freq += abs(letter_pair_frequencies[pair] - val) ** 2

    return num_of_hits, num_of_hits - sum_letter_freq * 10 - sum_pair_freq * 10 + 100


def select_parents(population, fitness_scores):
    return random.choices(population, fitness_scores, k=2)


def crossover(parent1, parent2):
    index = random.randint(0, 25)

    # put letters from parent1
    child = parent1.copy()
    letters_set = set(string.ascii_lowercase)

    # remove mapping, tot the keys!!!
    for ascii_value in range(ord('a'), ord('a') + index):
        letters_set.remove(parent1[chr(ascii_value)])

    # put letters from parent2
    for ascii_value in range(ord('a') + index, ord('z') + 1):
        if parent2[chr(ascii_value)] not in letters_set:
            child[chr(ascii_value)] = ""
        else:
            child[chr(ascii_value)] = parent2[chr(ascii_value)]
            letters_set.remove(parent2[chr(ascii_value)])

    # fix errors
    for key in child.keys():
        if child[key] == "":
            child[key] = letters_set.pop()

    return child


def mutate(mapping, big_muted):
    mute_rate = MUTATE_RATE
    letter_rate = LETTER_TO_CHANGE
    if big_muted:
        mute_rate = BIG_MUTATE_RATE
        letter_rate = BIG_LETTER_TO_CHANGE
    if random.random() >= mute_rate:
        return mapping
    letters = list(string.ascii_lowercase)
    for letter in mapping:
        if random.random() < letter_rate:
            letter_to_swap = random.randint(0, 25)
            while letters[letter_to_swap] == letter:
                letter_to_swap = random.randint(0, 25)
            tmp = mapping[letter]
            mapping[letter] = mapping[letters[letter_to_swap]]
            mapping[letters[letter_to_swap]] = tmp
    return mapping


def genetic_algorithm(ciphertext, dictionary, letter_frequencies, letter_pair_frequencies):
    global LEN_ENC
    LEN_ENC = len(ciphertext.lower().split())
    stack = [0] * 10
    population = [generate_random_mapping() for _ in range(POPULATION_SIZE)]
    best_mapping = None
    for generation in range(MAX_GENERATION):

        fitness_scores = []
        for mapping in population:
            decrypted_text = decrypt_text(ciphertext, mapping)
            _, fitness = calculate_fitness(decrypted_text, dictionary, letter_frequencies, letter_pair_frequencies)
            fitness_scores.append(fitness)

        avg_fitness = sum(fitness_scores) / POPULATION_SIZE
        best_fitness = max(fitness_scores)
        best_mapping = population[fitness_scores.index(best_fitness)]
        best_decrypt_text = decrypt_text(ciphertext, best_mapping)
        hit_rate, res = calculate_fitness(decrypt_text(ciphertext, best_mapping), dictionary, letter_frequencies, letter_pair_frequencies)

        BEST.append(best_fitness)
        AVG.append(avg_fitness)
        WORST.append(min(fitness_scores))


        stack[generation % 10] = hit_rate
        big_muted = False
        if sum(hit_rate == value for value in stack) == 7:
            big_muted = True
        if all(hit_rate == value for value in stack):
            if hit_rate / LEN_ENC < 0.7:
                return False, best_decrypt_text, best_fitness, best_mapping
            else:
                return True, best_decrypt_text, best_fitness, best_mapping

        print(str(hit_rate / LEN_ENC) + ", gen number is:" + str(generation))

        new_population = [mutate(best_mapping, big_muted)] * round(POPULATION_SIZE * NUM_BEST_TO_DUPLICATE)

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            new_population.append(mutate(crossover(parent1, parent2), big_muted))

        population = new_population

    return False, decrypt_text(ciphertext, best_mapping)


def main():
    ciphertext_file = 'enc.txt'
    dictionary_file = 'dict.txt'
    letter_frequencies_file = 'Letter_freq.txt'
    letter_pair_frequencies_file = 'Letter2_freq.txt'

    ciphertext = ''
    with open(ciphertext_file, 'r') as file:
        ciphertext = file.read().strip()

    dictionary = load_dictionary(dictionary_file)
    letter_frequencies = load_letter_frequencies(letter_frequencies_file)
    letter_pair_frequencies = load_letter_pair_frequencies(letter_pair_frequencies_file)

    start_time = time.time()
    best_fitness = 0
    best_mapping = {}
    best_decrypted_text = ''
    for i in range(10):
        solved, decrypted_text, fitness, mapping = genetic_algorithm(ciphertext, dictionary, letter_frequencies, letter_pair_frequencies)
        if fitness > best_fitness:
            best_fitness = fitness
            best_mapping = mapping
            best_decrypted_text = decrypted_text

        if solved:
            break

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print(best_decrypted_text)

    with open("plain.txt", 'w') as file:
        file.write(best_decrypted_text)
    with open("perm.txt", 'w') as file:
        for key, value in best_mapping.items():
            file.write(f"{key} {value}\n")
    print("The number of times the fitness function was called is: " + str(FITNESS_COUNT))


if __name__ == '__main__':
    for i in range(10):
        BEST = []
        AVG = []
        WORST = []
        print("**************************************" + str(i) + "*********************************************")
        main()
        # Saving data to CSV
        data = zip(BEST, AVG, WORST)
        csv_filename = 'scores.csv'

        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['BEST', 'AVG', 'WORST'])
            writer.writerows(data)

        print(f"The scores have been saved to {csv_filename}.")

        # Creating the graph
        x = range(1, len(BEST) + 1)

        plt.plot(x, BEST, label='BEST')
        plt.plot(x, AVG, label='AVG')
        plt.plot(x, WORST, label='WORST')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Regular Genetic algorithm - Scores Comparison')
        plt.legend()

        plt.show()
