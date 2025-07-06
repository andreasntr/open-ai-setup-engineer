from typing import Any
from json import dumps, load
from re import sub, fullmatch, IGNORECASE
from pandas import json_normalize
from numpy import argmin, array
from os.path import join
from cachetools.func import lru_cache

FRONT = 'Front'
REAR = 'Rear'
LEFT = 'Left'
RIGHT = 'Right'
LF = f'{LEFT}{FRONT}'
RF = f'{RIGHT}{FRONT}'
LR = f'{LEFT}{REAR}'
RR = f'{RIGHT}{REAR}'

DATA_PATH = 'data'
SETUP_SLUGS_DICT_PATH = join(DATA_PATH, 'setup_parts_slug_dict.json')
COMPONENTS_DICT = join(DATA_PATH, 'components_dict.json')


@lru_cache(maxsize=1024)
def get_slugs_dict() -> dict:
    """
    Gets the mapping between component technical description and its human-readable description.

    Returns:
        dict: Mapping between component technical description and its human-readable description.
    """
    with open(SETUP_SLUGS_DICT_PATH, 'r') as f:
        slugs_dict = load(f)
    return slugs_dict


@lru_cache(maxsize=1024)
def get_components_dict() -> dict:
    with open(COMPONENTS_DICT, 'r') as f:
        components_dict = load(f)
    return components_dict


def get_max_min(data: dict[str, Any]) -> tuple[float | int, float | int]:
    max_value = data.get('max')
    min_value = data.get('min')
    return max_value, min_value


def parse_setup_data(setup_json: dict[str, Any]):

    parsed_json = json_normalize(setup_json).iloc[0].to_dict()

    parsed_setup = {}

    for key, element in parsed_json.items():
        if isinstance(element, list):
            if isinstance(element[0], float):
                continue
            for key, value in parse_boundaries_list(element, key).items():
                if key not in parsed_setup.keys():
                    parsed_setup[key] = value
        else:
            verbose_key = key.replace('RF', RF).replace('LF', LF)
            parsed_setup[verbose_key] = {}
            parsed_setup[verbose_key] = element

    return parsed_setup


def parse_boundaries_list(element: dict[str, list[int]], key: str) -> dict:
    """
    Parses a dictionary of boundaries to a standard format.

    Args:
        element (dict[str, list[int]]): The dictionary of boundaries to parse.

    Returns:
        dict: The standardized boundaries dict.
    """
    boundaries = {}
    if len(element) == 2:
        boundaries[f'{key}.{FRONT}'] = {}
        boundaries[f'{key}.{REAR}'] = {}
        boundaries[f'{key}.{FRONT}'] = element[0]
        boundaries[f'{key}.{REAR}'] = element[1]
    elif len(element) == 4:
        boundaries[f'{key}.{LF}'] = {}
        boundaries[f'{key}.{RF}'] = {}
        boundaries[f'{key}.{LR}'] = {}
        boundaries[f'{key}.{RR}'] = {}
        boundaries[f'{key}.{LF}'] = element[0]
        boundaries[f'{key}.{RF}'] = element[1]
        boundaries[f'{key}.{LR}'] = element[2]
        boundaries[f'{key}.{RR}'] = element[3]
    return boundaries


def binarize(value: float | int, bins: list[int], bin_names: list[str]) -> str:
    """
    Fits a numeric value into a discrete, human-readable bin.

    Args:
        value (float | int): The value to binarize.
        bins (list[int): The bins to use for binarization.
        bin_names (list[str]): The names of the bins.

    Returns:
        str: The human-readable bin which the value belongs to.
    """
    # find the first index where bin >= value, i.e. where bin < value is False
    cmp_array = (array(bins) < value).astype(int)
    if cmp_array.min() == 1:
        return bin_names[-1]
    bin_name_index = argmin(cmp_array)
    return bin_names[bin_name_index]


def extract_setup_info(setup_json: dict, car_metadata: dict) -> str:
    """
    Turns a raw setup and its related car metadata into an LLM-readable description of the current user condition.

    Args:
        setup_json (dict): The setup to parse.
        car_metadata (dict): The car metadata to use for binarization.

    Returns:
        str: The LLM-readable description of the current user condition.
    """
    selected_car_metadata = car_metadata['coded_boundaries']
    selected_car_toe_camber_boundaries = car_metadata['actual_toe_camber_boundaries']
    parsed_setup = parse_setup_data(setup_json)

    parsed_setup = {
        sub(r'((basicSetup)|(advancedSetup))\.', '', key):
            {'value': parsed_setup[key]} |
            selected_car_metadata[key]
        for key in selected_car_metadata
    }

    setup_components = list(parsed_setup.keys())

    setup_analysis = {}

    for key in setup_components:

        value = parsed_setup[key]['value']
        min_clicks = parsed_setup[key]['min']
        max_clicks = parsed_setup[key]['max']
        toe_boundaries = {key.split('.')[-1].lower(): value
                          for key, value in selected_car_toe_camber_boundaries.items()
                          if fullmatch(r'.*(toe).*', key, flags=IGNORECASE)}
        camber_boundaries = {key.split('.')[-1].lower(): value
                             for key, value in selected_car_toe_camber_boundaries.items()
                             if fullmatch(r'.*(camber).*', key, flags=IGNORECASE)}
        if min_clicks == max_clicks:
            setup_analysis[key] = 'not available'
        elif fullmatch(r'.*(camber).*', key, flags=IGNORECASE):
            if 'front' in key.lower():
                max_value, min_value = get_max_min(camber_boundaries['front'])
            else:
                max_value, min_value = get_max_min(camber_boundaries['rear'])
            standard_bound = 4.0
            single_click_value = (max_value - min_value) / max_clicks
            # bring back into the [min_value, max_value] range
            value_in_degs = value * single_click_value
            if min_value != 0:
                value_in_degs -= min_value * (min_value/abs(min_value))
            steps = [
                -standard_bound, -standard_bound / 2, -standard_bound / 3,
                0,
                standard_bound / 3, standard_bound / 2, standard_bound,
                1000
            ]
            step_labels = ['highly negative', 'moderately negative', 'lightly negative',
                           'neutral',
                           'neutral', 'lightly positive', 'moderately positive',
                           'heavily positive']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = binarize(
                min_value,
                steps,
                step_labels)
            setup_analysis[key]['max'] = binarize(
                max_value,
                steps,
                step_labels)
            setup_analysis[key]['current'] = binarize(
                value_in_degs,
                steps,
                step_labels)
        elif fullmatch(r'.*(toe).*', key, flags=IGNORECASE):
            if 'front' in key.lower():
                max_value, min_value = get_max_min(toe_boundaries['front'])
            else:
                max_value, min_value = get_max_min(toe_boundaries['rear'])
            single_click_value = (max_value - min_value) / max_clicks
            standard_bound = 0.4
            single_click_value = (max_value - min_value) / max_clicks
            # bring back into the [min_value, max_value] range
            value_in_degs = value * single_click_value
            if min_value != 0:
                value_in_degs -= min_value * (min_value/abs(min_value))
            steps = [
                -standard_bound, -standard_bound / 2, -standard_bound / 3,
                0,
                standard_bound / 3, standard_bound/2, standard_bound,
                1000
            ]
            step_labels = ['high toe-in', 'moderate toe-in', 'light toe-in',
                           'neutral toe',
                           'neutral toe', 'light toe-out', 'moderate toe-out',
                           'heavy toe-out']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = binarize(
                min_value,
                steps,
                step_labels)
            setup_analysis[key]['max'] = binarize(
                max_value,
                steps,
                step_labels)
            setup_analysis[key]['current'] = binarize(
                value_in_degs,
                steps,
                step_labels)
        elif fullmatch(r'.*((arb)|(bump)|(rebound)).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 6
            step_labels = ['highly soft', 'moderately soft', 'lightly soft',
                           'lightly rigid', 'moderately rigid', 'heavily rigid']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2, config_step * 3,
                    config_step * 4, config_step * 5, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*(caster).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 4
            step_labels = ['low', 'moderate',
                           'high', 'very high']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*(steerratio).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 4
            step_labels = ['very high responsiveness', 'high responsiveness',
                           'linear responsiveness', 'low responsiveness']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*((tc1)|(tc2)|(abs)).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 5
            step_labels = ['extremely low', 'light',
                           'moderate',
                           'high', 'very high']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3,
                    config_step * 4, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*((wheelrate)|(bumpstoprate)).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 6
            step_labels = ['highly soft', 'moderately soft', 'lightly soft',
                           'lightly rigid', 'moderately rigid', 'heavily rigid']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2, config_step * 3,
                    config_step * 4, config_step * 5, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*(bumpstopwindow).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 5
            step_labels = ['very low suspension travel', 'small suspension travel',
                           'moderate suspension travel',
                           'high suspension travel', 'very high suspension travel']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3,
                    config_step * 4, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*(preload).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 3
            step_labels = ['light', 'moderate', 'high']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*(rideHeight).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 4
            step_labels = ['lowest', 'moderately low',
                           'moderately high', 'very high']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3, max_clicks
                ],
                step_labels)
        elif fullmatch(r'.*((splitter)|(rearwing)).*', key, flags=IGNORECASE):
            config_step = (max_clicks - min_clicks) / 5
            step_labels = ['lowest downforce', 'moderately low downforce',
                           'neutral',
                           'moderately high downforce', 'highest downforce']
            setup_analysis[key] = {}
            setup_analysis[key]['min'] = step_labels[0]
            setup_analysis[key]['max'] = step_labels[-1]
            setup_analysis[key]['current'] = binarize(
                value,
                [
                    min_clicks, config_step * 2,
                    config_step * 3,
                    config_step * 4, max_clicks
                ],
                step_labels)
    return prepare_setup_for_chatbot(setup_analysis)


def prepare_setup_for_chatbot(setup_json: dict) -> str:
    """
    Translates a standardized setup into an LLM-readable description of the same.

    Args:
        setup_json (dict): The setup to be translated.

    Returns:
        str: The LLM-readable description of the setup.
    """
    slugs_dict = get_slugs_dict()

    fields_to_drop = [
        slug for slug in setup_json if slug not in slugs_dict]
    for field in fields_to_drop:
        setup_json.pop(field)
    not_available_fields = []
    for field in setup_json:
        if setup_json[field] != 'not available':
            if setup_json[field]['current'] == setup_json[field]['max']:
                if field.startswith('alignment.camber') or \
                        field.startswith('alignment.toe'):
                    setup_json[field]['current'] += ' (can be more negative)'
                else:
                    setup_json[field]['current'] += ' (can be decreased)'
            elif setup_json[field]['current'] == setup_json[field]['min']:
                if field.startswith('alignment.camber') or \
                        field.startswith('alignment.toe'):
                    setup_json[field]['current'] += ' (can be more positive)'
                else:
                    setup_json[field]['current'] += ' (can be increased)'
            else:
                if field.startswith('alignment.camber') or \
                        field.startswith('alignment.toe'):
                    setup_json[field]['current'] += ' (can be more positive, can be more negative)'
                else:
                    setup_json[field]['current'] += ' (can be increased, can be decreased)'
            setup_json[field] = setup_json[field]['current']
        else:
            not_available_fields.append(field)
    for field in not_available_fields:
        setup_json.pop(field)
    setup_str = dumps(setup_json)

    for slug, replacement in slugs_dict.items():
        setup_str = setup_str.replace(slug, replacement)

    return setup_str


def remove_component_side(component: str) -> str:
    slugs_dict = get_slugs_dict()
    no_side_component = slugs_dict[component]
    if len(component.split()) == 3 and component.split()[1] in ['right', 'left']:
        no_side_component = f'{component.split()[0]} {component.split()[2]}'

    return no_side_component


def get_available_components(car_name: str) -> dict:
    return get_components_dict()[car_name]['available_components']


def get_unavailable_components(car_name: str) -> dict:
    return get_components_dict()[car_name]['unavailable_components']


def get_cars_with_same_unavailable_components(car_name: str) -> list[str]:
    """
    Returns the car names that have the same unavailable components as the given car.

    Args:
        car_name (str): The name of the car to compare with.

    Returns:
        list[str]: A list of cars with the same unavailable components
    """
    return get_components_dict()[car_name]['cars_with_same_unavailable']
