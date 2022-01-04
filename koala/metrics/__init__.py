import math

from textdistance import levenshtein, hamming


def levenshtein_similarity(word_1, word_2):
    """Levenshtein similarity"""
    return levenshtein.normalized_similarity(word_1.upper(), word_2.upper())


def hamming_similarity(word_1, word_2):
    """hamming similarity"""
    return hamming.normalized_similarity(word_1.upper(), word_2.upper())


def norm_distances(len_1, errors):
    """Returns the normalized Levenshtein distance given the length and the errors.

    This functions is thought to be applied to the fuzzy counts of a fuzzy regex match. This returns three values:
    (replacements, insertions, deletions). The distance is the sum of them. To compute the normalized version, it must
    be divided by the maximum length of both strings. One is given, the other one is given by
    len_1 + insertions - deletions.
    """
    len_2 = len_1 + errors[1] - errors[2]
    return sum(errors) / max(len_1, len_2)


def compute_max_number_errors_levenshtein(phrase: str, minimum_similarity: float) -> int:
    """Unnormalizes the levenshtein distance.
    Givena levenshtein distance, computes the maximum number of errors a string must have to make a match
    with a given phrase.
    """
    max_error = math.ceil((1 - minimum_similarity) * len(phrase))
    max_error = max(max_error, 1)
    return max_error
