import math
import random
import string
import re
from collections import Counter


alphabet = list(string.ascii_lowercase)
mutation_rate = 0.1
Letter_Freq = None
Letter2_Freq = None
dict_words = None
enc = None
number_of_generations = 80
number_in_generations = 1000
letter_count = {}
letter2_count = {}
h = 0
h2 = 0
count = 0


def txt_to_dict(file_path):
    """Reads a text file and returns a dictionary mapping each one to its frequency."""
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 2:
                key = line[1].lower()
                value = float(line[0])
                result_dict[key] = value
    return result_dict


def txt_to_hash(file_path):
    """Reads a text file and returns a set of all words in the file."""
    result_hash = set()
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()
            if word != '':
                result_hash.add(word)
    return result_hash


def txt_to_string(file_path):
    """Reads a text file and returns a string containing all the text."""
    str_ = ''
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()
            str_ += word + '\n'
    return str_


def generate_letter_mapping():
    """Generates a random mapping from letters to letters."""
    shuffled_alphabet = list(alphabet)
    random.shuffle(shuffled_alphabet)
    letter_mapping = dict(zip(alphabet, shuffled_alphabet))
    return letter_mapping


def combine_dictionaries(dict1, dict2):
    """Combines two dictionaries, splitting them at a random letter."""
    split_point = random.choice(alphabet)
    combined_dict = {}
    for key in alphabet:
        if key <= split_point:
            combined_dict[key] = dict1[key]
        else:
            combined_dict[key] = dict2[key]
    return combined_dict


def mutation(dictionary):
    """Mutates a dictionary by swapping two random letters."""
    if random.random() <= mutation_rate:
        keys = list(dictionary.keys())
        if len(keys) >= 2:
            letter1, letter2 = random.sample(keys, 2)
            dictionary[letter1], dictionary[letter2] = dictionary[letter2], dictionary[letter1]
    return dictionary


def choose(dict_list):
    """Chooses two dictionaries from a list."""
    list_of_best = []
    odds = []
    for dict_ in dict_list:
        odds.append(evaluate_dict(dict_))
    s = sum(odds)
    cumulative_odds = [sum(odds[:i + 1]) for i in range(len(odds))]
    for _ in range(len(dict_list)):
        random_number = random.uniform(0, s)
        random_number1 = random.uniform(0, s)
        choosen = get_chosen_dict(random_number, cumulative_odds, dict_list)
        choosen1 = get_chosen_dict(random_number1, cumulative_odds, dict_list)
        list_of_best.append((choosen, choosen1))
    return list_of_best


def get_chosen_dict(random_number, cumulative_odds, dict_list):
    """Returns the dictionary with that chosen."""
    for i, odds in enumerate(cumulative_odds):
        if random_number <= odds:
            return dict_list[i]


def count_letters():
    """Counts the number of occurrences of each letter in a string."""
    text = enc
    counter = 0
    for letter in text:
        if letter.isalpha():
            counter += 1
            letter = letter.lower()
            if letter in letter_count:
                letter_count[letter] += 1
            else:
                letter_count[letter] = 1

    for key in letter_count.keys():
        letter_count[key] = letter_count[key] / counter


def count2_letters():
    """Counts the number of occurrences of each pair of letters in a string."""
    text = enc
    counter = 0
    for letter1, letter2 in zip(text[:-1], text[1:]):
        if letter1.isalpha() and letter2.isalpha():
            counter += 1
            letter1 = letter1.lower()
            letter2 = letter2.lower()
            concat = letter1 + letter2
            if concat in letter2_count:
                letter2_count[concat] += 1
            else:
                letter2_count[concat] = 1

    for key in letter2_count.keys():
        letter2_count[key] = letter2_count[key] / counter


def evaluate_dict(dict_):
    """Evaluates a dictionary by comparing it to a set of expected frequencies."""
    global h, h2, count
    count += 1
    p = [0.5, 0.5]
    a = evaluate_by_letter(dict_)
    b = evaluate_by_letters(dict_)
    hit = evaluate_by_word(dict_)
    h2 += hit
    if hit < h + 3:
        return 0
    return math.pow(a, 5) * p[0] + math.pow(b, 3) * p[1]


def evaluate_dict2(dict_):
    """Evaluates a dictionary by comparing it to a set of expected frequencies."""
    global count
    count += 1
    p = [0.5, 0.5]
    a = evaluate_by_letter(dict_)
    b = evaluate_by_letters(dict_)
    return math.pow(a, 5) * p[0] + math.pow(b, 3) * p[1]


def evaluate_by_letter(dict_):
    """Evaluates a dictionary by comparing it to a set of expected frequencies."""
    sum_ = 0
    a = {}
    for key, value in letter_count.items():
        a[dict_[key]] = value
    for key, value in a.items():
        sum_ += abs(value - Letter_Freq[key])
    return 1 - sum_


def evaluate_by_letters(dict_):
    """Evaluates a dictionary by comparing it to a set of expected frequencies."""
    sum_ = 0
    a = {}
    for key, value in letter2_count.items():
        key2 = dict_[key[0]] + dict_[key[1]]
        a[key2] = value
    for key, value in a.items():
        sum_ += abs(value - Letter2_Freq[key])
    return 1 - sum_


def evaluate_by_word(dict_):
    """Evaluates a dictionary by counting the number of words it can decode."""
    enc2 = enc
    enc3 = ''
    for char in enc2:
        if char.isalpha():
            enc3 += dict_[char]
        else:
            enc3 += char
    words = re.findall(r'\b\w+\b', enc3)
    k = 0
    setwords = set(words)
    for word in setwords:
        if word in dict_words:
            k += 1
    return k


def fix(dict_):
    """Fixes a dictionary by making sure all letters are unique."""
    lista = []
    free = []
    for key, value in dict_.items():
        if value not in lista:
            lista.append(value)
        else:
            free.append(key)
    remaining = list(dict_.keys() - lista)
    for i in range(len(free)):
        dict_[free[i]] = remaining[i]
    return dict_


def preventing_early():
    global h, mutation_rate
    mutation_rate = 1
    h = 0


def main():
    dar = False
    lam = False
    generation = []
    global h, h2, mutation_rate, count
    h3 = 0
    for _ in range(0, number_in_generations):
        generation.append(generate_letter_mapping())
    for j in range(1, number_of_generations + 1):
        generation2 = []
        mutation_rate = 0.1
        print(h, "   ", h3)
        if abs(h - h3) < 1 and j < 0.8 * number_of_generations:
            preventing_early()
        for i in choose(generation):
            temp = fix(combine_dictionaries(i[0], i[1]))
            if lam:
                mutation_rate = 1
                dorween = mutation(temp)
                if evaluate_by_word(dorween) > evaluate_by_word(temp):
                    temp = dorween
            elif dar:
                pass
            else:
                mutation_rate = 0.1
                temp = mutation(temp)
            generation2.append(temp)
        generation = generation2
        s = sum(evaluate_by_letter(k) for k in generation)
        s2 = sum(evaluate_by_letters(k) for k in generation)
        print(j, " - ", s/number_in_generations, ', ', s2/number_in_generations)
        h3 = h
        h = h2 / number_in_generations
        h2 = 0
        print(h)
        if j % 20 == 0:
            best = (0, 0)
            l = ''
            for k in generation:
                if evaluate_by_word(k) > best[1]:
                    best = (k, evaluate_by_word(k))
            for z in enc:
                if z.isalpha():
                    l += best[0][z]
                else:
                    l += z
            print(best[1])
            print(l)

    best = (0, 0)
    for k in generation:
        if evaluate_by_word(k) > best[1]:
            best = (k, evaluate_by_word(k))
    for z in enc:
        if z.isalpha():
            l += best[0][z]
        else:
            l += z
    print(l)
    print(count)

if __name__ == "__main__":
    Letter_Freq = txt_to_dict("Letter_Freq.txt")
    Letter2_Freq = txt_to_dict("Letter2_Freq.txt")
    dict_words = txt_to_hash("dict.txt")
    enc = txt_to_string("enc.txt")
    count_letters()
    count2_letters()
    main()

