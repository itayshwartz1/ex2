import random
import string
import time
from data_utils import Data_utils


POPULATION_SIZE = 100
WORDS_WEIGHT = 0.33
SELECTED_SIZE = 15
MUTATION_RATE = 0.05
GENERATIONS = 300
LOCAL_MAXIMUM = 10
STOP_NO_CHANGE = 30
TOO_MUCH_GENERATIONS = 100
STARTERS = 5
LAMARC = 5
DARWIN = 2


class GeneticAlgorithm:
    def __init__(self):
        self.data_utils = Data_utils()
        self.two_letters = self.data_utils.get_letters2_freq()
        self.one_letter = self.data_utils.get_letters_freq()
        self.common_words = set(self.data_utils.get_words())
        self.encrypted_text = self.data_utils.get_encrypted_text()
        self.chars = [char for char in string.ascii_lowercase]
        self.population = self.generate_population()
        self.best_dict = None
        self.best_generation = 0
        self.best_fitness = float('-inf')
        self.stop = 0
        self.local_maximum = 0
        self.count_iters = 0

    def generate_population(self):
        list_of_dicts = []
        while len(list_of_dicts) < POPULATION_SIZE:
            chars_copy = self.chars.copy()
            random.shuffle(chars_copy)
            dictionary = {self.chars[i]: chars_copy[i] for i in range(len(self.chars))}
            list_of_dicts.append(dictionary)
        return list_of_dicts

    def text_decoder(self, dictionary):
        decoded_text = ""
        for letter in self.encrypted_text:
            decoded_text += dictionary.get(letter, letter)
        return decoded_text

    def one_letter_fitness(self, text):
        chars = [c for c in string.ascii_lowercase]
        one_letter_freq = {char: text.count(char) / len(text) for char in chars}
        sum_one_letter = sum([pow(self.one_letter[char] - one_letter_freq.get(char, 0), 2) for char in chars])
        return sum_one_letter

    def two_letter_fitness(self, text):
        chars = [c for c in string.ascii_lowercase]
        len_text = len(text) - 1
        two_letter_freq = {}
        for i in range(len_text):
            two_letter = text[i: (i + 2)]
            if two_letter in self.two_letters:
                temp = int(two_letter_freq.get(two_letter, 0))
                two_letter_freq[two_letter] = temp + 1
        sum_all_two_letters = sum(two_letter_freq.values())
        two_letter_new_freq = {two_letters: val / sum_all_two_letters for two_letters, val in two_letter_freq.items()}
        sum_two_letters = sum(
            [pow(self.two_letters[char] - two_letter_new_freq.get(char, 0), 2) for char in self.two_letters])
        return sum_two_letters

    def words_fitness(self, text):
        count_words = 0
        for line in text.split("/n"):
            for word in line.split(" "):
                clean_word = word.lower().rstrip(",.").rstrip()
                if clean_word in self.common_words:
                    count_words += 1
        return pow(count_words * 0.3, 2)

    def fitness(self, dictionary):
        self.count_iters += 1
        text = self.text_decoder(dictionary)
        return self.one_letter_fitness(text) + self.two_letter_fitness(text) + self.words_fitness(text)

    def selection(self, best_fitness):
        samples = random.sample(best_fitness, SELECTED_SIZE)
        sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)
        index, fitness = sorted_samples[0]
        return self.population[index]

    def correct_dict(self, dictionary):
        chars = [c for c in string.ascii_lowercase]
        letters_in_dict = set(dictionary.values())
        not_in_dict = list(set(chars) - set(letters_in_dict))
        values_in_use = set()
        for letter in dictionary:
            if dictionary[letter] in values_in_use:
                for i in range(len(not_in_dict)):
                    if not_in_dict[i] not in values_in_use:
                        dictionary[letter] = not_in_dict[i]
                        values_in_use.add(not_in_dict[i])
                        break
            else:
                values_in_use.add(dictionary[letter])
        return dictionary

    def crossover(self, first_parent, second_parent):
        cutoff = random.choice(list(first_parent.keys()))
        first_child = {key: first_parent[key] if key <= cutoff else second_parent[key] for key in first_parent}
        second_child = {key: second_parent[key] if key <= cutoff else first_parent[key] for key in second_parent}
        return self.correct_dict(first_child), self.correct_dict(second_child)

    def mutation(self, dictionary):
        for i in range(len(dictionary)):
            if random.random() < MUTATION_RATE:
                sample_first, sample_second = random.sample(range(len(self.chars)), 2)
                key_first, key_second = self.chars[sample_first], self.chars[sample_second]
                dictionary[key_first], dictionary[key_second] = dictionary[key_second], dictionary[key_first]
        return dictionary

    def mutate(self, dictionary):
        keys = random.sample(list(dictionary.keys()), 2)
        dictionary[keys[0]], dictionary[keys[1]] = dictionary[keys[1]], dictionary[keys[0]]
        return dictionary

    def lamarc(self, dictionary):
        copy_dict = dictionary.copy()
        fitness_val = self.fitness(dictionary)
        for i in range(LAMARC):
            swapped_dict = self.mutate(copy_dict)
            new_fitness = self.fitness(swapped_dict)
            if new_fitness > fitness_val:
                copy_dict = swapped_dict
                fitness_val = new_fitness
        return copy_dict

    def darwin(self, dictionaries):
        copy_dicts = dictionaries.copy()
        best_dict = None
        fitness = self.fitness(copy_dicts[0])
        for dictionary in copy_dicts:
            for i in range(DARWIN):
                dictionary = self.mutate(dictionary)
                new_fitness_val = self.fitness(dictionary)
                if new_fitness_val > fitness:
                    best_dict = dictionary
                    fitness = new_fitness_val
        return best_dict, fitness

    def next_generation(self, lamrac, darwin):
        for generation in range(POPULATION_SIZE):
            fitness_scores = sorted([(i, int(self.fitness(dictionary))) for i, dictionary in
                                     enumerate(self.population)], key=lambda x: x[1], reverse=True)
            index_, fitness = fitness_scores[0]
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_generation = generation
                self.best_dict = self.population[index_]
                self.stop = 0
                self.local_maximum = 0

            parents = [self.selection(fitness_scores) for _ in range(POPULATION_SIZE)]
            children = []
            len_parents = len(parents)
            for x in range(0, POPULATION_SIZE):
                parent1 = parents[x % len_parents]
                parent2 = parents[(x+1) % len_parents]
                child1, child2 = self.crossover(parent1, parent2)
                children.append(self.mutation(child1))
                children.append(self.mutation(child2))
                x += 2

            if lamrac:
                for x in range(POPULATION_SIZE):
                    children[x] = self.lamarc(children[x])

            if darwin:
                best_darwin_dict, best_fitness_darwin = self.darwin(children)
                if best_fitness_darwin > self.best_fitness:
                    self.best_fitness = best_fitness_darwin
                    self.best_dict = best_darwin_dict
                    self.best_generation = generation
                    self.stop = 0
                    self.local_maximum = 0

            if self.best_dict not in children:
                children[0] = self.best_dict

            self.stop += 1
            self.population = children

            if self.local_maximum == LOCAL_MAXIMUM:
                print("In local maximum")
                return self.best_dict, self.best_generation, self.best_fitness

            self.local_maximum += 1
            if self.stop == STOP_NO_CHANGE and generation > TOO_MUCH_GENERATIONS:
                print("No change")
                break

        print("Go over all generations")
        return self.best_dict, self.best_generation, self.best_fitness


if __name__ == '__main__':
    start_time = time.time()
    outputs = []
    algorithm = None
    i = 0
    while STARTERS > 0:
        i += 1
        print("run number: " + str(i))
        algorithm = GeneticAlgorithm()
        outputs.append(algorithm.next_generation(lamrac=True, darwin=False))
        print("Best fitness is - " + str(algorithm.best_fitness))
        print("Best generation is - " + str(algorithm.best_generation))
        print("Number of fitness calls is - " + str(algorithm.count_iters))
        print("-----------------------------")
        STARTERS -= 1

    best_fitness = float('-inf')
    best_dict_all = None
    best_generation = 0
    for best_dict, generation, fitness in outputs:
        if fitness > best_fitness:
            best_fitness = fitness
            best_dict_all = best_dict
            best_generation = generation

    chars = [c for c in string.ascii_lowercase]
    values_list = list(best_dict_all.values())
    with open("perm.txt", "w") as file:
        for i, symbol in enumerate(chars):
            file.write(f"{symbol}: {values_list[i]}\n")
    file.close()
    with open("plain.txt", "w") as file:
        file.write(algorithm.text_decoder(best_dict_all))
    file.close()

    time_end = time.time()
    all_time = (time_end - start_time) / 60
    print("Time for algorithm: ", all_time)
