from tqdm import tqdm
from pathlib import Path
from pickle import load as pickle_load
from glob import glob as glob_glob
from os.path import join as os_path_join
from datetime import datetime
from typing import Tuple
from re import compile as re_compile, DOTALL as re_DOTALL, MULTILINE as re_MULTILINE
from json import loads as json_loads
from ast import literal_eval as ast_literal_eval
from httpx import Response


strptime = datetime.strptime
PATTERN_MARKDOWN_PYTHON = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
REGEX_COMPILED = re_compile(PATTERN_MARKDOWN_PYTHON, re_DOTALL | re_MULTILINE)


def response_text_to_list_dicts(response_text: str) -> list[dict]:
    text_list_cleaned = []
    for paragraph in response_text.split('\n'):
        if paragraph.startswith('data:'):
            string = f'{{{paragraph.replace('data:', '"data":').replace('[DONE]', '"[DONE]"')}}}'
            try:
                cleaned_paragraph = json_loads(string)
            except:
                print(string)
            text_list_cleaned.append(cleaned_paragraph)

    return text_list_cleaned


def find_video_timeframes(string) -> list[str]:
    '''
    _summary_

    Args:
        string (str): the content of a response data entry

    Returns:
        list[str]: the list of video timestamps

    Example:
        >>> find_video_timeframes(text_list_cleaned[1]['data']['choices'][0]['messages'][0]['delta']['content'])
        ['00:00:01',
         '00:00:07',
         '00:00:11',
         '00:00:16',
         '00:00:21',
         '00:00:24',
         '00:00:31',
         '00:00:35',
         '00:00:40',
         '00:00:52',
         '00:00:55']
    '''
    # regex = r'^(?:\d+(?::[0-5][0-9]:[0-5][0-9])?|[0-5]?[0-9]:[0-5][0-9])$' # doesn't work.
    regex = r'\d{2}:\d{2}:\d{2}' # works
    # works by removing the start and end
    # regex = r'(?:\d+(?::[0-5][0-9]:[0-5][0-9])?|[0-5]?[0-9]:[0-5][0-9])'
    timeframes = re_compile(regex).findall(string)
    return timeframes


# It's slightly different
# {'Is the woman concerned with the man': ['00:00:12', '00:00:17', '00:00:47'],
#  'Is the woman excited to see the man?': ['00:00:12',
#                                           '00:00:17',
#                                           '00:00:47',
#                                           '00:00:59'],
#  'What kind of laugh does the man let out?': ['00:00:12', '00:00:47'],
#  'Why does the woman keep looking behind her?': ['00:00:12',
#                                                  '00:00:17',
#                                                  '00:00:47'],
#  'Why does the woman put one finger up?': ['00:00:12']}


def process_success_response(success_response) -> Tuple[str, list[str]]:
    success_response_data_dicts = response_text_to_list_dicts(
        success_response.text)
    try:
        assert len(success_response_data_dicts) >= 2
    except AssertionError as e:
        raise ValueError(f'len(success_response_data_dicts)={len(
            success_response_data_dicts)}. success_response_data_dicts={success_response_data_dicts}') from e
    content_1 = success_response_data_dicts[1]['data']['choices'][0]['messages'][0]['delta']['content']
    # question is where to extract the question
    question = str(success_response.request.body).split(
        '\\"')[1].split('\\')[0]
    timestamps = find_video_timeframes(content_1)
    return question, timestamps


# for each video:
def process_video(video_dict: dict, indices_or_timestamps: str) -> dict[list[str]]:
    '''
    _summary_

    Args:
        video_dict (dict): _description_

    Raises:
        AssertionError: _description_

    Returns:
        dict: {question: [timeframes]}
    '''
    return_dict = {}
    # for each success (question), get all data entries
    for success_response in video_dict['successes']:
        # for each data entry, get all possible frames
        question, timestamps = process_success_response(success_response)
        frame_indices = [timestamp2index(
            timestamp, '0.00-60.019000', 100) for timestamp in timestamps]
        if indices_or_timestamps == 'indices':
            return_dict[question] = frame_indices
        else:
            return_dict[question] = timestamps
    return return_dict


# def map_timeframes_to_index(timeframes: list[str]):
# timeframes (list[str]): ['00:00:00'-'00:00:55', ...]

def timestamp2float(timestamp: str) -> float:
    '''
    _summary_

    Args:
        timestamp (str): '0:00:55'

    Returns:
        _type_: _description_
    '''
    return strptime(timestamp, '%H:%M:%S').second


def timestamp2index(timestamp: str, timeframes_floats_start_end: str, num_frames: int) -> int:
    '''
    '00:00:55' => 16

    Args:
        timestamp (str): '00:00:55' from GPT4V, etc.
        timeframe_duration (str): '0.00-60.019000' from the dataset
        num_fames (int): The number of frames. 100 for video-chatgpt. Unsure for Azure.

    Raises:
        AssertionError: _description_

    Returns:
        int: the frame index
    '''
    timestamp_float = timestamp2float(timestamp)
    start_str, end_str = timeframes_floats_start_end.split('-')
    start_float, end_float = float(start_str), float(end_str)
    duration = end_float - start_float
    segment_seconds = duration / num_frames
    frame_index = (timestamp_float - start_float) // segment_seconds
    return int(frame_index)


def get_gpt4v_responses(dirpath: str) -> list[dict]:
    '''
    _summary_

    Args:
        a (int): _description_
        c (list, optional): _description_. Defaults to [1,2].

    Raises:
        AssertionError: _description_

    Returns:
        _type_: _description_
    '''
    response_dicts = {}
    responses_fnames = glob_glob(os_path_join(dirpath, '*.pkl'))
    for response_fname in tqdm(responses_fnames):
        with open(response_fname, 'rb') as f:
            response_dict = pickle_load(f)
            response_dicts[Path(response_fname).stem] = response_dict
            if len(response_dict) != 4:
                print(response_fname, len(response_dict))

    return response_dicts  # But then how to deal with so many "data"?


def extract_python_string_from_text(text: str) -> str:
    return REGEX_COMPILED.search(text).groups()[0]


def httpx_response2text(response: Response) -> str:
    return response.json()['choices'][0]['message']['content']


def httpx_response2dict(response: Response) -> Tuple[dict, str]:
    str_dict = extract_python_string_from_text(httpx_response2text(response))
    return ast_literal_eval(str_dict), str_dict


def get_max_timestamps(response_dict: dict[str, dict]) -> list[str]:
    '''
    _summary_

    Args:
        response_dict (dict[str, dict]): _description_

    Raises:
        AssertionError: _description_

    Returns:
        list[str]: _description_
    '''
    relevance_max = max(response_dict.values())
    return [k for k, v in response_dict.items() if v == relevance_max]
