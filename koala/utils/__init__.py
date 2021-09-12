import regex


def fuzzy_matcher(phrase: str, text: str, minimum_similarity: float, maximum_number_errors=5000):
    """Returns a list of fuzzy matches using levenshtein distance
    If maximum_number_errors is indicated, spans with more errors are filtered. Useful when searched phrase is
    very long."""
    max_errors = min(compute_max_number_errors_levenshtein(phrase, minimum_similarity), maximum_number_errors)
    regex_expression = "(" + phrase + ")" + "{e<" + str(max_errors) + "}"

    regex_expression = regex.compile(regex_expression, regex.BESTMATCH | regex.IGNORECASE)
    matches = list(regex.finditer(regex_expression, text))
    return matches
