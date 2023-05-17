'''
Itay Shwarts 318528171
Noa Eitan 316222777
'''

import re, random, string, copy
NUM_OF_SOLUTIONS = 20

class Problem:
    def __init__(self):
        self.words = []
        #TO DO
        self.letter_freq = {}
        self.letter2_freq = {}


class Solution:
    def __init__(self):
        self.perm = {}
        self.words = []
        self.fitness = 0

    def calculate_fitness(self):
        pass

    def init_words(self, words):
        permuted_words = []
        for word in words:
            permuted_word = ''
            for letter in word:
                permuted_letter = self.perm.get(letter.lower())
                if permuted_letter is None:
                    permuted_word += letter.lower()
                else:
                    permuted_word += permuted_letter
            permuted_words.append(permuted_word)
        self.words = permuted_words


    def init_perm(self, perm):
        self.perm = perm

def read_words_from_file():
    with open('enc.txt', 'r') as file:
        content = file.read()
        words = re.findall(r'\b\w+\b', content)
        return words




def init_population(words):
    solutions = []
    letters = list(string.ascii_lowercase)
    for i in range(NUM_OF_SOLUTIONS):
        letters_copy = letters.copy()
        random.shuffle(letters_copy)
        sol = Solution()
        sol.init_perm(dict(zip(string.ascii_lowercase, letters_copy)))
        sol.init_words(words)
        solutions.append(sol)

    return solutions


def main():
    problem = Problem()
    problem.words = read_words_from_file()
    population = init_population(problem.words)


if __name__ == '__main__':
    main()
