from unittest import TestCase

from koala.utils.number_processing import normalize_numbers


class Test(TestCase):
    def test_normalize_numbers_long_string_form(self):
        dict_numbers = {"ocho millones cuatrocientos doce mil once": 8_412_011,
                        "cien mil ocho": 100_008}
        for s, f in dict_numbers.items():
            self.assertEqual(normalize_numbers(s), f)
