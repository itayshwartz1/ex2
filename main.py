'''
Itay Shwartz 318528171
Noa Eitan 316222777
'''

import re, random, string, copy

NUM_OF_SOLUTIONS = 50
MAX_GENERATION = 1000
PERCENT_TO_DUPLICATE = 0.2
NUM_OF_LETTERS = 26
PERCENTE_TO_MUTATE = 0.5

class Problem:
    def __init__(self):
        self.letter_freq = {}
        self.letter2_freq = {}
        self.dict_words = []

        with open('enc.txt', 'r') as file:
            content = file.read()
            self.words = re.findall(r'\b\w+\b', content)

        self.num_of_words_in_enc = len(self.words)

        with open('Letter_Freq.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    number, letter = line.split()
                    self.letter_freq[letter.lower()] = float(number)

        with open('Letter2_Freq.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    number, letters = line.split()
                    self.letter2_freq[letters.lower()] = float(number)
                else:
                    break

        with open('dict.txt', 'r') as file:
            self.dict_words = [line.strip() for line in file if line.strip() != ""]


class Solution:
    def __init__(self, perm):
        self.perm = perm
        self.words = []
        self.fitness = 0

    def set_perm(self, perm):
        self.perm = perm

    def init_words(self, enc_words):
        permuted_words = []
        for word in enc_words:
            permuted_word = ''
            for letter in word:
                permuted_letter = self.perm.get(letter.lower())
                if permuted_letter is None:
                    permuted_word += letter.lower()
                else:
                    permuted_word += permuted_letter
            permuted_words.append(permuted_word)
        self.words = permuted_words


    def do_mutate(self):
        letters = list(string.ascii_lowercase)
        index1 = random.randint(0, NUM_OF_LETTERS - 1)
        index2 = random.randint(0, NUM_OF_LETTERS - 1)
        while index1 == index2:
            index2 = random.randint(0, NUM_OF_LETTERS - 1)

        tmp = self.perm[letters[index1]]
        self.perm[letters[index1]] = self.perm[letters[index2]]
        self.perm[letters[index2]] = tmp


def calculate_fitness_of_individuals(problem, population):
    letter_freq_array_dis = []
    letter2_freq_array_dis = []
    num_of_hits_array = []

    for solution in population:
        num_of_hits = 0

        # initial dict of letters, all the value are 0
        ascii_dict = {ascii_value: 0 for ascii_value in string.ascii_lowercase}

        # initial dict of pairs of letters, all the value are 0
        pair_dict = {}
        for letter1 in string.ascii_lowercase:
            for letter2 in string.ascii_lowercase:
                pair = letter1 + letter2
                pair_dict[pair] = 0

        # check hits, for each letter we count the number of appearance, same to pairs
        for word in solution.words:
            if word in problem.dict_words:
                num_of_hits += 1
            for letter in word:
                ascii_dict[letter] += 1
            for i in range(len(word) - 1):
                pair = word[i: i + 2]
                pair_dict[pair] += 1

        num_of_hits_array.append(num_of_hits)

        # normalize the count of appearance of letters to percentage
        sum_of_letters = sum(ascii_dict.values())

        sum_letter_freq = 0
        for letter in ascii_dict:
            ascii_dict[letter] /= sum_of_letters
            sum_letter_freq += abs(problem.letter_freq[letter] - ascii_dict[letter])
        letter_freq_array_dis.append(sum_letter_freq)

        # normalize the count of appearance of pairs of letters to percentage
        sum_of_pairs = sum(pair_dict.values())
        sum_pair_freq = 0
        for pair in pair_dict:
            pair_dict[pair] /= sum_of_pairs
            sum_pair_freq += abs(problem.letter2_freq[pair] - pair_dict[pair])
        letter2_freq_array_dis.append(sum_pair_freq)

    total_letter_freq = sum(letter_freq_array_dis)
    total_letter2_freq = sum(letter2_freq_array_dis)

    for i, solution in enumerate(population):
        solution.fitness = 0
        solution.fitness += (num_of_hits_array[i] / problem.num_of_words_in_enc) * 0.7
        solution.fitness += (letter_freq_array_dis[i] / total_letter_freq) * 0.2
        solution.fitness += (letter2_freq_array_dis[i] / total_letter2_freq) * 0.1


def init_population(problem):
    solutions = []
    for i in range(NUM_OF_SOLUTIONS):
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        sol = Solution(dict(zip(string.ascii_lowercase, letters)))
        sol.init_words(problem.words)
        solutions.append(sol)
    calculate_fitness_of_individuals(problem, solutions)

    return solutions


def select_parent(population):
    # Extract the fitness values from the solutions
    fitness_values = [solution.fitness for solution in population]

    # Randomly choose a solution based on the weights
    selected_solution = random.choices(population, weights=fitness_values)[0]

    return selected_solution


def crossover(problem, parent1, parent2):
    index = random.randint(0, NUM_OF_LETTERS - 1)

    # put letters from parent1
    new_perm = parent1.perm.copy()
    letters_set = set(string.ascii_lowercase)

    # remove mapping, tot the keys!!!
    for ascii_value in range(ord('a'), ord('a') + index):
        letters_set.remove(parent1.perm[chr(ascii_value)])

    # put letters from parent2
    for ascii_value in range(ord('a') + index, ord('z') + 1):
        if parent2.perm[chr(ascii_value)] not in letters_set:
            new_perm[chr(ascii_value)] = ""
        else:
            new_perm[chr(ascii_value)] = parent2.perm[chr(ascii_value)]
            letters_set.remove(parent2.perm[chr(ascii_value)])

    # fix errors
    for key in new_perm.keys():
        if new_perm[key] == "":
            new_perm[key] = letters_set.pop()
    sol = Solution(new_perm)
    sol.init_words(problem.words)
    return sol


def get_fitness(solution):
    return solution.fitness


def mutate(new_population):
    number_of_mutations = round(PERCENTE_TO_MUTATE * NUM_OF_SOLUTIONS)
    random_array = random.sample(range(0, NUM_OF_SOLUTIONS - 1), number_of_mutations)

    for index in random_array:
        new_population[index].do_mutate()


# 5% of the new population will be the best 5% of the old population
# 95% crossover
# 20% of the new population will be mutate
def next_gen(problem, population):
    new_population = []

    population.sort(key=get_fitness, reverse=True)
    print("the best fitness is:" + str(population[0].fitness))
    num_of_best_sol = round(PERCENT_TO_DUPLICATE * NUM_OF_SOLUTIONS)
    new_population = population[:num_of_best_sol]

    while len(new_population) < NUM_OF_SOLUTIONS:
        parent1 = select_parent(population)
        parent2 = select_parent(population)

        child = crossover(problem, parent1, parent2)
        new_population.append(child)

    mutate(new_population)
    calculate_fitness_of_individuals(problem, new_population)
    return new_population


def write_sol(problem, population):
    pass


def genetic_algorithm(problem, population):
    generation = 0
    while generation < MAX_GENERATION:  # max, min avg
        population = next_gen(problem, population)
        generation += 1
    write_sol(problem, population)
    print("Number of generations is:" + str(generation))


def main():
    problem = Problem()
    population = init_population(problem)
    genetic_algorithm(problem, population)


if __name__ == '__main__':
    main()
