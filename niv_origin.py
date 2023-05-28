import random
import string

# Constants
POPULATION_SIZE = 100
TOURNAMENT_SELECTION_RATE = 0.2
MUTATION_RATE = 0.2
NUM_GENERATION = 200
PROBABILITY = 0.4
RESULTS_FOR_NEXT = 0.1
HIT_RATE = 0


def get_index(letter):
    letter = letter.upper()  # Convert the letter to uppercase

    if len(letter) != 1 or not letter.isalpha():
        return None  # Return None for invalid input

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    index = alphabet.index(letter) + 1  # Add 1 to make the index 1-based

    return index


def load_enc(filename):
    # Load letter frequencies from the given file
    encrypted_text = ""
    with open(filename, "r") as file:
        encrypted_text = file.read()
    return encrypted_text.upper()


def load_letter_frequencies(filename):
    # Load letter frequencies from the given file
    frequencies = {}
    with open(filename, "r") as file:
        for line in file:
            frequency, letter = line.strip().split()
            frequencies[letter] = float(frequency)
    return frequencies


def load_letter2_frequencies(filename):
    # Load letter frequencies from the given file
    frequencies = {}
    with open(filename, "r") as file:
        for line in file:
            frequency, letter = line.strip().split()
            frequencies[letter] = float(frequency)
            if letter == 'ZZ':
                return frequencies
    return frequencies


def load_dictionary(filename):
    # Load a dictionary of valid English words from the given file
    dictionary = set()
    with open(filename, "r") as file:
        for line in file:
            word = line.strip()
            dictionary.add(word)
    return dictionary


def generate_initial_population():
    # Generate an initial population of random substitution tables
    population = []
    for _ in range(POPULATION_SIZE):
        keys = list(string.ascii_uppercase)
        table = list(string.ascii_uppercase)
        random.shuffle(table)
        while any(table[i] == table[i + 1] for i in range(25)):
            random.shuffle(table)
        population.append(table)
    return population


def check_alphabet_list(chars):
    unique_chars = [char for char in chars if chars.count(char) == 1]
    alphabet = [char for char in chars if chars.count(char) > 1]
    alphabet_set = set(alphabet)
    missing_chars = [char for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if char not in chars]
    for letter in alphabet_set:
        missing_chars.append(letter)
    random.shuffle(missing_chars)
    result = []
    for char in chars:
        if char in unique_chars:
            result.append(char)
        elif char in alphabet:
            result.append(missing_chars.pop(0).upper())

    return result


def decrypt_text(encrypted_text, substitution_table):
    decrypted_text = []
    for char in encrypted_text:
        if char.upper() in substitution_table:
            decrypted_text.append(substitution_table[get_index(char) - 1])
        else:
            decrypted_text.append(char)
    return ''.join(decrypted_text)


def crossover(parent1, parent2):
    # Perform crossover between two parents to produce a child
    # In this example, we use a simple one-point crossover
    if random.random() < PROBABILITY:
        crossover_point = random.randint(1, len(parent1) - 2)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        child1 = check_alphabet_list(child1)
        child2 = check_alphabet_list(child2)
        return child1, child2
    return parent1, parent2


def mutate(table):
    # Perform mutation on the substitution table
    # In this example, we swap two random letters
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(table)), 2)
        table[idx1], table[idx2] = table[idx2], table[idx1]
    return table


def select_parents(sorted_dic):
    # Select two parents from the population using tournament selection
    #tournament_size = int(POPULATION_SIZE * TOURNAMENT_SELECTION_RATE)
    tournament_size = 10
    tournament = random.sample(sorted_dic, tournament_size)
    parent1 = max(tournament, key=lambda x: x[1])
    tournament.remove(parent1)
    parent2 = max(tournament, key=lambda x: x[1])
    return parent1[0], parent2[0]


def fitness(decrypted_text):
    hit_rate, fitness_freq, fitness_2freq = 0, 0, 0
    letter_counts = {letter: 0 for letter in string.ascii_lowercase}
    letters = string.ascii_lowercase
    two_letter_combinations = [a + b for a in letters for b in letters]
    letter2_counts = {combination: 0 for combination in two_letter_combinations}
    for word in decrypted_text:
        new_word = word.replace(",", "").replace(".", "").replace("/", "").replace(":", "").replace(";","").rstrip().lstrip().lower()
        if new_word in dictionary:
            hit_rate += 1
        for i in range(len(new_word)):
            if new_word[i].isalpha():
                letter_counts[new_word[i]] += 1
            if i == len(new_word) - 1:
                break
            if new_word[i].isalpha() and new_word[i + 1].isalpha():
                letter2_counts[new_word[i:i + 2]] += 1
    hit_rate = hit_rate/len(decrypted_text)
    for item in letter_counts:
        fitness_freq += abs(letter_counts[item] * letter_frequencies[item.upper()])
    for item in letter2_counts:
        fitness_2freq += abs(letter2_counts[item] * letter2_frequencies[item.upper()])
    return (hit_rate + fitness_freq + fitness_2freq), hit_rate

def genetic_algorithm(encrypted_text):
    global MUTATION_RATE, PROBABILITY
    s, f = set(), set()
    best_table = []
    population = generate_initial_population()
    hit_rate_counter, max_fitness, max_hit_rate, steps, local_max = 0, 0, 0, 0, 0
    while steps < NUM_GENERATION and max_hit_rate != 1:
        res_fitness_dict = {}
        fitness_arr, new_population = [], []
        for res in population:
            decrypted_text = decrypt_text(encrypted_text, res)
            fitness_res, hit_rate = fitness(decrypted_text.split())
            fitness_arr.append(fitness_res)
            if fitness_res > max_fitness:
                max_fitness = fitness_res
                max_hit_rate = hit_rate
        if max_hit_rate in s:
            hit_rate_counter += 1
        else:
            if s != set():
                s.pop()
            s.add(max_hit_rate)
            hit_rate_counter = 0

        # calculating the best fitness and table
        best_fitness = max(fitness_arr)
        best_table = population[fitness_arr.index(best_fitness)]
        print("the best fitness of iteration ", steps, "is ", best_fitness)
        print("the best hitrate is ", max_hit_rate)
        # building the local maximum
        if best_fitness in f:
            local_max += 1
        else:
            if f != set():
                f.pop()
            f.add(best_fitness)
            local_max = 0
        # checking the local maximum
        if local_max == int(0.05 * POPULATION_SIZE) or hit_rate_counter == int(0.05 * POPULATION_SIZE):
            print("CHANGED THE MUTATION RATE TO 0.9")
            MUTATION_RATE = 1
            PROBABILITY = 1
        res_fitness_dict = list(zip(population, fitness_arr))
        res_fitness_dict = sorted(res_fitness_dict, key=lambda x: x[1], reverse=True)
        for item in range(int(0.1 * POPULATION_SIZE)):
            new_population.append(best_table)
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(res_fitness_dict)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population[:POPULATION_SIZE]
        if local_max % int(0.1 * POPULATION_SIZE) == 2:
            print("CHANGED BACK THE MUTATION RATE TO 0.2")
            PROBABILITY = 0.4
            MUTATION_RATE = 0.2
        steps += 1

    best_decrypted_text = decrypt_text(encrypted_text, best_table)
    # Write the decrypted text to plain.txt
    with open("plain.txt", "w") as file:
        file.write(best_decrypted_text.lower())

    # Write the substitution table to perm.txt
    with open("perm.txt", "w") as file:
        for i, letter in enumerate(string.ascii_uppercase):
            file.write(f"{letter} -> {best_table[i]}\n")

    print(f"Number of steps: {steps}")


# Load letter frequencies and dictionary
letter_frequencies = load_letter_frequencies("Letter_Freq.txt")
letter2_frequencies = load_letter2_frequencies("Letter2_Freq.txt")
dictionary = load_dictionary("dict.txt")

# Encrypted text to be decrypted
encrypted_text = load_enc("enc.txt")

# Run the genetic algorithm
genetic_algorithm(encrypted_text)
