import random
import string

# Constants
POPULATION_SIZE = 100
TOURNAMENT_RATE = 0.2
TOURNAMENT_SIZE = 10
MUTATION_RATE = 0.2
MAX_GEN = 200
CROSSOVER_RATE = 0.4
RESULTS_FOR_NEXT = 0.1
HIT_RATE = 0
LAST_PAIR = 'ZZ'
LETTERS = string.ascii_lowercase

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
            # real letter
            decrypted_text.append(mapping[letter_to_index(letter) - 1])
        else:
            # not real letter
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


def fitness(decrypted_text):
    global FITNESS_STEPS
    FITNESS_STEPS += 1

    hit_rate = 0
    letter_freq = 0
    letter2_freq = 0

    letter_counts = {letter: 0 for letter in string.ascii_lowercase}
    letters = string.ascii_lowercase
    two_letter_combinations = [a + b for a in letters for b in letters]
    letter2_counts = {combination: 0 for combination in two_letter_combinations}
    for word in decrypted_text:
        new_word = word.replace(",", "").replace(".", "").replace("/", "").replace(":", "").replace(";","").rstrip().lstrip().lower()
        if new_word in DICT:
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
        letter_freq += abs(letter_counts[item] * LETTER_FREQ[item.lower()])
    for item in letter2_counts:
        letter2_freq += abs(letter2_counts[item] * LETTER2_FREQ[item.lower()])
    return (hit_rate + letter_freq + letter2_freq), hit_rate


def genetic_algorithm(encrypted_text):
    global MUTATION_RATE, CROSSOVER_RATE
    hit_rate_counter, max_fitness, max_hit_rate, local_max = 0, 0, 0, 0
    s, f = set(), set() # max localy
    population = create_population()
    best_mapping = []
    for _ in range(MAX_GEN):
        fitness_scores = []
        new_population = []
        for child in population:
            decrypted_text = decrypt_text(encrypted_text, child)
            fitness_res, hit_rate = fitness(decrypted_text.split())
            fitness_scores.append(fitness_res)
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
        best_fitness = max(fitness_scores)
        best_mapping = population[fitness_scores.index(best_fitness)]
        print("the best hitrate is ", max_hit_rate)
        # building the local maximum
        if best_fitness in f:
            local_max += 1
        else:
            if f != set():
                f.pop()
            f.add(best_fitness)
            local_max = 0
        # checking the local maximum. כמה דורות אני נמצא באותו ערך אז אני משנה את המוטציות
        if local_max == int(0.05 * POPULATION_SIZE) or hit_rate_counter == int(0.05 * POPULATION_SIZE):
            print("CHANGED THE MUTATION RATE TO 0.9")
            MUTATION_RATE = 1
            CROSSOVER_RATE = 1
        res_fitness_dict = list(zip(population, fitness_scores))
        res_fitness_dict = sorted(res_fitness_dict, key=lambda x: x[1], reverse=True)
        for item in range(int(0.1 * POPULATION_SIZE)):
            new_population.append(best_mapping)
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = get_parents(res_fitness_dict)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population[:POPULATION_SIZE]
        if local_max % int(0.1 * POPULATION_SIZE) == 2:
            print("CHANGED BACK THE MUTATION RATE TO 0.2")
            CROSSOVER_RATE = 0.4
            MUTATION_RATE = 0.2

    best_decrypted_text = decrypt_text(encrypted_text, best_mapping)
    # Write the decrypted text to plain.txt
    with open("plain.txt", "w") as file:
        file.write(best_decrypted_text.lower())

    # Write the substitution table to perm.txt
    with open("perm.txt", "w") as file:
        for i, letter in enumerate(string.ascii_lowercase):
            file.write(f"{letter} {best_mapping[i]}\n")

    print(f"Number of steps: {FITNESS_STEPS}")


genetic_algorithm(ENC_TXT)
