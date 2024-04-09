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
from pprint import pformat
from warnings import warn
from httpx import Response
from azure_video_qa import PROMPT_BEFORE, PROMPT_AFTER, _load_vqa_file


strptime = datetime.strptime
PATTERN_MARKDOWN_PYTHON = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
REGEX_COMPILED = re_compile(PATTERN_MARKDOWN_PYTHON, re_DOTALL | re_MULTILINE)


# for each video:
def process_video_dict(video_id: str, video_dict: dict, indices_or_timestamps: str) -> dict[list[str]]:
    '''
    _summary_

    Args:
        video_dict (dict): _description_

    Raises:
        AssertionError: _description_

    Returns:
        dict: {question: [timeframes]}
    '''
    full_ts = '0.00-60.019000' # TODO load full ts from qa gt file
    return_dict = {}
    failures = []
    # for each success (question), get all data entries
    for success_response in video_dict['successes']:
        # for each data entry, get all possible frames
        question = extract_question_from_response_request(success_response.request)
        try:
            dict_frame_relevance, str_frame_relevance = httpx_response2dictstr(success_response)
        except ValueError as e:
            warn(f'ValueError encountered while processing video_id={video_id}, question={question}')
            failures.append(question)
            continue
        timestamps = get_max_timestamps(dict_frame_relevance)
        if indices_or_timestamps == 'indices':
            frame_indices = [timestamp2index(timestamp, full_ts, 100) for timestamp in timestamps]
            return_dict[question] = frame_indices
        else:
            print(question, timestamps)
            return_dict[question] = timestamps

    return return_dict, failures


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
    timestamp = timestamp[:12]
    if '.' in timestamp:
        format_string = '%H:%M:%S.%f'
    else:
        format_string = '%H:%M:%S'
    return strptime(timestamp, format_string).second


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
    match = REGEX_COMPILED.search(text)
    if match is None:
        raise ValueError(f'No match found in text={text}')
    return match.groups()[0]


def httpx_response2text(response: Response) -> str:
    response_json = response.json()
    try:
        return response_json['choices'][0]['message']['content']
    except KeyError as e:
        raise ValueError(f'ValueError encountered while processing response_json=\n\n{pformat(response_json, indent=2)}') from e


def httpx_response2dictstr(response: Response) -> Tuple[dict, str]:
    text = httpx_response2text(response)
    if text == '':
        raise ValueError(f'No text extracted for response=\n\n{pformat(response.json(), indent=2)}. \n\nrequest={pformat(jsonify_request(response.request), indent=2)}')
    str_dict = extract_python_string_from_text(text)
    return ast_literal_eval(str_dict), str_dict


def get_max_timestamps(dict_frame_relevance: dict[str, dict]) -> list[str]:
    '''
    _summary_

    Args:
        dict_frame_relevance (dict[str, dict]): _description_

    Raises:
        AssertionError: _description_

    Returns:
        list[str]: _description_
    '''
    relevance_max = max(dict_frame_relevance.values())
    return [k for k, v in dict_frame_relevance.items() if v == relevance_max]


def jsonify_request(request):
    my_bytes_value = request.body.decode()
    dict_request = json_loads(my_bytes_value)
    return dict_request


def extract_question_from_response_request(request):
    dict_request = jsonify_request(request)
    text = dict_request['messages'][-1]['content'][-1]['text']
    return text[text.index(PROMPT_BEFORE)+len(PROMPT_BEFORE):text.index(PROMPT_AFTER)]


# video: question: ts, etc
def process_videos_dicts():
    dirpath = 'result'
    response_dicts = get_gpt4v_responses(dirpath)
    dict_video_id_question_ts_successes = {}
    dict_video_id_question_ts_failures = {}
    for video_id, response_dict in response_dicts.items():
        successes, failures = process_video_dict(video_id, response_dict, 'indices')
        dict_video_id_question_ts_successes[video_id] = successes
        if failures:
            dict_video_id_question_ts_failures[video_id] = failures
    return dict_video_id_question_ts_successes, dict_video_id_question_ts_failures


def add_ts_to_gt(fpath_qa, ts_dict):
    dict_video_qa = _load_vqa_file(fpath_qa)
    failures_dict = {}
    for video_id, questions in dict_video_qa.items():
        if video_id not in ts_dict:
            continue

        failures = []
        for question in questions:
            question_text = question['q']
            question_text = question_text.replace('Why does the man in the brown shirt snickwer at :36?', 'Why does the man in the brown shirt snigger at :36?')
            question_text = question_text.replace('Why does the African American man stutter at 0:41', 'Why does the black man stutter at 0:41')

            try:
                ts_dict_question = ts_dict[video_id][f'"{question_text}"']
            except KeyError:
                print(video_id, f'"{question_text}"')
                failures.append(f'"{question_text}"')
                continue
            question['gt_frame'] = ts_dict_question
        if failures:
            failures_dict[video_id] = failures
    return dict_video_qa


def load_and_add_ts_to_gt(fpath_qa):
    successes, _ = process_videos_dicts()
    return add_ts_to_gt(fpath_qa, successes)
