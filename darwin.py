import random
import string
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
TOURNAMENT_RATE = 0.2
TOURNAMENT_SIZE = 10
MUTATION_RATE = 0.2
MAX_GEN = 500
CROSSOVER_RATE = 0.4
RESULTS_FOR_NEXT = 0.1
HIT_RATE = 0
LAST_PAIR = 'ZZ'
LETTERS = string.ascii_lowercase
LETTER_APPEARANCE = {letter: 0 for letter in LETTERS}
LETTER2_APPEARANCE = {combination: 0 for combination in [i + j for i in LETTERS for j in LETTERS]}
LOCAL_MAX_RATE = int(0.05 * POPULATION_SIZE)
ELITISM = 0.1
MAX_POWER_MODE = 12
N = 3


global AVG
AVG = []

global BEST
BEST = []

global WORST
WORST = []

global FITNESS_STEPS
FITNESS_STEPS = 0


def read_enc(filename):
    with open(filename, "r") as file:
        return file.read().lower().strip()


def read_letter_freq(filename):
    freq = {}
    with open(filename, "r") as file:
        for line in file:
            f, l = line.strip().split()
            freq[l.lower()] = float(f)
    return freq


def read_letter2_freq(filename):
    freq = {}
    with open(filename, "r") as file:
        for line in file:
            f, l2 = line.strip().split()
            freq[l2.lower()] = float(f)
            if l2 == LAST_PAIR:
                break
    return freq


def read_dict(filename):
    dictionary = set()
    with open(filename, "r") as file:
        for line in file:
            if line.strip() != "":
                dictionary.add(line.strip().lower())
    return dictionary


LETTER_FREQ = read_letter_freq("Letter_Freq.txt")
LETTER2_FREQ = read_letter2_freq("Letter2_Freq.txt")
DICT = read_dict("dict.txt")
ENC_TXT = read_enc("enc.txt")
LEN_ENC = len(ENC_TXT.split())


def create_population():
    population = []
    for i in range(POPULATION_SIZE):
        child = list(string.ascii_lowercase)
        random.shuffle(child)
        population.append(child)
    return population


def fix_child(child):
    letter_to_add = set()
    unique_letters = set()
    for letter in LETTERS:
        count = child.count(letter)
        if count > 1 or count == 0:
            letter_to_add.add(letter)
        else:
            unique_letters.add(letter)

    fixed_child = []
    for char in child:
        if char in unique_letters:
            fixed_child.append(char)
        else:
            fixed_child.append(letter_to_add.pop())

    return fixed_child


def letter_to_index(letter):
    return LETTERS.index(letter) + 1


def decrypt_text(enc_text, mapping):
    decrypted_text = []
    for letter in enc_text:
        if letter in mapping:
            decrypted_text.append(mapping[letter_to_index(letter) - 1])
        else:
            decrypted_text.append(letter)
    return ''.join(decrypted_text)


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        index = random.randint(1, len(parent1) - 2)
        return fix_child(parent1[:index] + parent2[index:]), fix_child(parent2[:index] + parent1[index:])
    return parent1, parent2


def mutate(mapping):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(26), 2)
        mapping[i], mapping[j] = mapping[j], mapping[i]
    return mapping


def get_parents(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    parent1 = max(tournament, key=lambda x: x[1])
    tournament.remove(parent1)
    parent2 = max(tournament, key=lambda x: x[1])
    return parent1[0], parent2[0]


def remove_chars(word):
    chars_to_remove = ".;,:\n"
    cleaned_string = word.translate(str.maketrans('', '', chars_to_remove))
    return cleaned_string.rstrip().lstrip().lower()

def fitness(decrypted_text):
    global FITNESS_STEPS
    FITNESS_STEPS += 1
    letter_appearance = LETTER_APPEARANCE.copy()
    pairs_appearance = LETTER2_APPEARANCE.copy()
    hit_rate = 0
    letter_freq = 0
    letter2_freq = 0

    for word in decrypted_text:
        new_word = remove_chars(word)
        if new_word in DICT:
            hit_rate += 1
        size = len(new_word)
        for i in range(size):
            letter_appearance[new_word[i]] += 1
            if i == size - 1:
                break
            pair = new_word[i:i + 2]
            pairs_appearance[pair] += 1

    hit_rate = hit_rate/LEN_ENC
    for letter in letter_appearance:
        letter_freq += letter_appearance[letter] * LETTER_FREQ[letter]
    for pair in pairs_appearance:
        letter2_freq += pairs_appearance[pair] * LETTER2_FREQ[pair]
    return (hit_rate + letter_freq + letter2_freq), hit_rate

def local_optimization(original_mapping, origin_fitness):
    new_mapping = original_mapping.copy()
    for _ in range(N):
        i = random.randint(0, 25)
        j = random.randint(0, 25)
        while i == j:
            j = random.randint(0, 25)

        tmp = new_mapping[i]
        new_mapping[i] = new_mapping[j]
        new_mapping[j] = tmp

    new_fitness, _ = fitness(decrypt_text(ENC_TXT, new_mapping))
    if new_fitness > origin_fitness:
        return new_mapping, new_fitness
    return original_mapping, origin_fitness

def genetic_algorithm(encrypted_text):

    global MUTATION_RATE, CROSSOVER_RATE
    hit_rate_counter = 0
    max_fitness = 0
    current_max_hit_rate = 0
    best_fitness_counter = 0
    prev_max_hit_rate = 0
    prev_best_fitness = 0
    population = create_population()
    best_mapping = []

    for gen in range(MAX_GEN):
        fitness_scores = []
        for child in population:
            decrypted_text = decrypt_text(encrypted_text, child)
            fitness_res, hit_rate = fitness(decrypted_text.split())
            _, new_fitness = local_optimization(child, fitness_res)
            fitness_scores.append(new_fitness)
            if fitness_res > max_fitness:
                max_fitness = fitness_res
                current_max_hit_rate = hit_rate

        current_best_fitness = max(fitness_scores)
        best_mapping = population[fitness_scores.index(current_best_fitness)]

        if current_max_hit_rate == prev_max_hit_rate:
            hit_rate_counter += 1
        else:
            prev_max_hit_rate = current_max_hit_rate
            hit_rate_counter = 0

        BEST.append(current_best_fitness)
        AVG.append(sum(fitness_scores) / POPULATION_SIZE)
        WORST.append(min(fitness_scores))

        print("the best hitrate is ", current_max_hit_rate)
        if current_best_fitness == prev_best_fitness:
            best_fitness_counter += 1
        else:
            prev_best_fitness = current_best_fitness
            best_fitness_counter = 0
        if best_fitness_counter == LOCAL_MAX_RATE or hit_rate_counter == LOCAL_MAX_RATE:
            MUTATION_RATE = 1
            CROSSOVER_RATE = 1


        population_with_fitness = sorted(list(zip(population, fitness_scores)), key=lambda x: x[1], reverse=True)
        new_population = [best_mapping] * round(ELITISM * POPULATION_SIZE)

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = get_parents(population_with_fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:POPULATION_SIZE]
        if best_fitness_counter == MAX_POWER_MODE:
            CROSSOVER_RATE = 0.4
            MUTATION_RATE = 0.2

        if current_max_hit_rate > 0.5:
            if best_fitness_counter == 15 or hit_rate_counter == 15 or current_max_hit_rate == 1:
                break

    best_text = decrypt_text(encrypted_text, best_mapping)
    with open("plain.txt", "w") as file:
        file.write(best_text)

    with open("perm.txt", "w") as file:
        for i, letter in enumerate(LETTERS):
            file.write(letter + " " + best_mapping[i] + "\n")

    print("The number of calls to fitness function is: " + str(FITNESS_STEPS))

    x = range(1, len(BEST) + 1)

    plt.plot(x, BEST, label='BEST')
    plt.plot(x, AVG, label='AVG')
    plt.plot(x, WORST, label='WORST')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Darwin Genetic algorithm - Scores Comparison')
    plt.legend()

    plt.show()

genetic_algorithm(ENC_TXT)


