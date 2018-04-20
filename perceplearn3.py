import sys
import re

NUMBER_OF_CLASSES = 2


def learn_model():
    weight_vector = initialize_weights()
    bias = initialize_bias()
    training_data_path = sys.argv[1]
    training_data = open(training_data_path, 'r', encoding='utf-8')
    learn_model_parameters(training_data, weight_vector, bias)
    print(weight_vector[0])
    print(weight_vector[1])
    print(bias)


def learn_model_parameters(training_data, weight_vector, bias):
    count_vector = {}
    for line in training_data:
        words = line.split(" ")
        for word in get_word_list(words):
            add_one(count_vector, filter_word(word))
        for index, classification in enumerate(get_classification(words)):
            activation = calculate_activation(count_vector, weight_vector[index], bias[index])
            if activation * get_sign(classification) <= 0:
                update_weights(weight_vector[index], get_sign(classification), count_vector)
                bias[index] += get_sign(classification)
        count_vector.clear()


def update_weights(weight_vector, sign, count_vector):
    for word, freq in count_vector.items():
        new_weight = sign * freq
        if word in weight_vector:
            new_weight += weight_vector[word]
        weight_vector[word] = new_weight


def calculate_activation(count_vector, weight_vector, bias):
    activation = 0
    for word, count in count_vector.items():
        if word in weight_vector:
            activation += count * weight_vector[word]
    activation += bias
    return activation


def get_word_list(words):
    return words[NUMBER_OF_CLASSES + 1:]


def get_classification(word_list):
    classification = []
    for index in range(1, NUMBER_OF_CLASSES + 1):
        classification.append(word_list[index])
    return classification


def add_one(dictionary, key):
    if key not in dictionary:
        dictionary[key] = 1
    else:
        dictionary[key] += 1


def initialize_weights():
    weight_vector = {}
    for index in range(0, NUMBER_OF_CLASSES):
        weight_vector[index] = {}
    return weight_vector


def initialize_bias():
    bias = {}
    for index in range(0, NUMBER_OF_CLASSES):
        bias[index] = 0
    return bias


def get_sign(classification):
    if classification == "True" or classification == "Pos":
        return 1
    else:
        return -1


def filter_word(word):
    punctuations = re.compile('([-!.,"()=/\\\])')
    word = re.sub(punctuations, "", word)
    return word


if __name__ == "__main__":
    learn_model()