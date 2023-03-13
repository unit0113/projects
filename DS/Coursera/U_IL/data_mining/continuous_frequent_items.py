import os
import sys
from collections import defaultdict


MIN_SUPPORT = 0.01


def read_data():
    path = os.path.join(sys.path[0], 'reviews_sample.txt')

    raw_samples = []
    with open(path, 'r') as file:
        for line in file.readlines():
            raw_samples.append(line.strip())

    return raw_samples


def get_n_grams(raw_samples, length):
    n_gram_dict = defaultdict(lambda: 0)
    
    for line in raw_samples:
        line_list = line.split(' ')
        added_n_grams = []
        for index in range(len(line_list) - length + 1):
            n_gram = ' '.join(line_list[index:index+length])
            if n_gram not in added_n_grams:
                n_gram_dict[n_gram] += 1
                added_n_grams.append(n_gram)

    return n_gram_dict


def get_frequent(n_grams, num_samples):
    req_support = int(MIN_SUPPORT * num_samples)
    return {key: value for key, value in n_grams.items() if value >= req_support}


def write_freq(freq_n_grams, write_path):
    with open(write_path, 'a') as file:
        for key, value in sorted(freq_n_grams.items(), key=lambda kv: kv[1]):
            file.write(f'{value}:{key.replace(" ", ";")}\n')


def get_freq_cont(raw_samples):
    write_path = os.path.join(sys.path[0], 'cont_patterns.txt')
    if os.path.exists(write_path):
        os.remove(write_path)

    num_samples = len(raw_samples)
    n_gram_len = 1

    n_grams = get_n_grams(raw_samples, n_gram_len)
    freq_n_grams = get_frequent(n_grams, num_samples)

    while freq_n_grams:
        write_freq(freq_n_grams, write_path)
        n_gram_len += 1
        n_grams = get_n_grams(raw_samples, n_gram_len)
        freq_n_grams = get_frequent(n_grams, num_samples)


if __name__ == '__main__':
    raw_samples = read_data()
    get_freq_cont(raw_samples)
