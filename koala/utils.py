from typing import List, Dict

from koala.abstracts import Sample


def convert_to_samples(list_of_data: List[Dict], field_text="text",
                       field_label="label") -> List[Sample]:
    results = [Sample(x[field_text], x[field_label]) for x in list_of_data]
    return results


def split_samples_per_label(list_of_samples: List[Sample], ) -> Dict:
    dict_results = {}
    for s in list_of_samples:
        dict_results[s.label] = dict_results.get(s.label, []) + [s]
    return dict_results
