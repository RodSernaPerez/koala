import re

from functools import reduce

dict_name_to_number = {"uno": 1,
                       "dos": 2,
                       "tres": 3,
                       "cuatro": 4,
                       "cinco": 5,
                       "seis": 6,
                       "siete": 7,
                       "ocho": 8,
                       "nueve": 9,
                       "diez": 10,
                       "once": 11,
                       "doce": 12,
                       "trece": 13,
                       "catorce": 14,
                       "quince": 15,
                       "dieciseis": 16,
                       "diecisiete": 17,
                       "dieciocho": 18,
                       "diecinueve": 19,
                       "veinte": 20,
                       "veintiuno": 21,
                       "veintidos": 22,
                       "veintitres": 23,
                       "veinticuatro": 24,
                       "veinticinco": 25,
                       "veintiseis": 26,
                       "veintisiete": 27,
                       "veintiocho": 28,
                       "veintinueve": 29,
                       "treinta": 30,
                       "cuarenta": 40,
                       "cincuenta": 50,
                       "sesenta": 60,
                       "setenta": 70,
                       "ochenta": 80,
                       "noventa": 90,
                       "cien": 100,
                       "ciento": 100,
                       "doscientas": 200,
                       "trescientas": 300,
                       "cuatrocientas": 400,
                       "quinientas": 500,
                       "seiscientas": 600,
                       "setecientas": 700,
                       "ochocientas": 800,
                       "novecientas": 900,
                       "doscientos": 200,
                       "trescientos": 300,
                       "cuatrocientos": 400,
                       "quinientos": 500,
                       "seiscientos": 600,
                       "setecientos": 700,
                       "ochocientos": 800,
                       "novecientos": 900
                       }

dict_multiplicadores = {
    "centesima": 0.01,
    "centesimas": 0.01,
    "milesimas": 0.001,
    "mil": 1000,
    "miles": 1000,
    "millon": 1000000,
    "millones": 1000000}

list_splitters_integer_decimal_parts = ["unidad",
                                        "unidades",
                                        "entero",
                                        "enteros",
                                        "con"]


def normalize_numbers(string):
    """
    Converts the string format of a number to a float.
    Considers 3 cases:
        - Numbers like 3.215.238,28 should be cleaned (remove dots and replace comma by a dot).
        - Numbers like 3.215.238 28 should be cleaned and reformat (typically, the comma is not detected by the ocr)
        - Numbers like 'tres millones doscientos quince mil doscientos treinta y ocho' need a more complex logic."""
    try:
        # First case: only replace commas and dots
        text_cleaned = string.replace(".", "").replace(",", ".")
        s = float(text_cleaned)
    except ValueError:
        text_cleaned = string.replace(".", "")
        if re.match(r"^[0-9]+\s[0-9]{2}$", text_cleaned):
            text_cleaned = text_cleaned[:-3] + "." + text_cleaned[-2:]
            s = float(text_cleaned)
        elif re.match(r"[0-9]", string):  # At this point should not be any digit in the string
            # Unknown format
            s = string
        else:
            s = _number_string_to_digits(string)
    return s


def _number_string_to_digits(string):
    """Given the string representation of a number, returns it as an integer.
    Example:
        >> number_string_to_digits("DIEZ MILLONES OCHENTA Y CINCO MIL CIENTO DIECIOCHO")
        10085118
    """

    def convert_tokens_to_number(tokens):
        cont = dict_name_to_number.get(tokens[0], 1)
        tokens = tokens[1:]

        if len(tokens):
            if tokens[0] == "y":
                tokens = tokens[1:]
            cont = cont + convert_tokens_to_number(tokens)
        return cont

    def convert_part(string):
        x = []
        cont = 0
        for n in string.split():
            if n in dict_multiplicadores.keys():
                if not len(x):
                    x = ["un"]
                value = convert_tokens_to_number(x)
                value = value * dict_multiplicadores[n]
                cont = cont + value
                x = []
            else:
                x.append(n)

        if len(x):
            cont = cont + convert_tokens_to_number(x)
        return cont

    string = string.lower()
    string = reduce(lambda s, x:
                    s.replace(x[0], x[1]), [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), (",", "")],
                    string)
    # To split into integer, decimal part
    parts = []
    n = []
    for x in string.split():
        if x in list_splitters_integer_decimal_parts:
            parts.append(" ".join(n))
            n = []
        else:
            n.append(x)
    parts.append(" ".join(n))
    cont = reduce(lambda x, y: x + y, [convert_part(p) for p in parts])

    return cont
