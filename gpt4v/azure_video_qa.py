
from os import access as os_access, W_OK as os_W_OK
from os.path import exists as os_path_exists, join as os_path_join, basename as os_path_basename, dirname as os_path_dirname, isdir as os_path_isdir
from glob import glob as glob_glob
from json import load as json_load, dump as json_dump
from collections import defaultdict
from pathlib import Path
from time import sleep
from pickle import dump as pickle_dump
from httpx import get as httpx_get
from requests import post as requests_post, put as requests_put, delete as requests_delete, RequestException
from tqdm.notebook import tqdm  # pip install ipywidgets
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI


TIMEOUT = 100
PROMPT_BEFORE = 'How relevant is each video frame to answering the question: '
PROMPT_AFTER = '? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question. For example: {"00:00:06.0050000": 0.0, "00:00:15.0130000": 0.02, "00:00:16.0130000": 0.03, "00:00:17.0140000": 0.08, "00:00:18.0150000": 0.07, "00:00:19.0160000": 0.02, "00:00:20.0170000": 0.03, "00:00:22.0180000": 0.05, "00:00:24.0200000": 0.09, "00:00:30.0250000": 0.10, "00:00:31.0260000": 0.99, "00:00:37.0310000": 0.81, "00:00:38.0320000": 0.77, "00:00:39.0330000": 0.75, "00:00:40.0330000": 0.79, "00:00:41.0340000": 0.65, "00:00:43.0360000": 0.62, "00:00:48.0400000": 0.22, "00:00:49.0410000": 0.09, "00:00:50.0000000": 0.11}'


def get_unique_videos(list_of_dicts: list[dict], vid_column_name: str) -> dict:
    video_qa_dict = defaultdict(list)

    for qa in list_of_dicts:
        video_qa_dict[qa[vid_column_name]].append(qa)
    return video_qa_dict


def _load_vqa_file(video_qa_json_fpath: str) -> dict:
    '''
    _summary_

    Args:
        None
    Raises:
        AssertionError: _description_

    Returns:
        dict: a dictionary that looks like video, [questions]
    '''
    assert os_path_exists(video_qa_json_fpath)
    with open(video_qa_json_fpath, 'r') as file_in:
        list_of_dicts = json_load(file_in)
    dict_video_qa = get_unique_videos(list_of_dicts, 'vid_name')
    return dict_video_qa


def get_video_ingestion_status(response_json, video_id_with_i):
    dict_ingestion = {entry['name']: entry['state'] for entry in response_json['value']}
    if video_id_with_i in dict_ingestion:
        return dict_ingestion[video_id_with_i]


class AzureVideoQA:
    def __init__(self, config):
        '''
        Requires credentials

        Args:
            config (CfgNode): _description_

        Raises:
            AssertionError: _description_

        Returns:
            _type_: _description_
        '''
        # 1. Set up local files
        self.dirpath_videos = dirpath_videos = config.LOCAL_FILES.DIRPATH_VIDEOS
        assert os_path_exists(dirpath_videos), f'dirpath_video does not exist: {dirpath_videos}'
        self.fpath_qa_json_train = fpath_qa_json_train = config.LOCAL_FILES.FPATH_QA_JSON_TRAIN
        assert os_path_exists(fpath_qa_json_train), f'dirpath_video does not exist: {fpath_qa_json_train}'
        self.fpath_qa_json_val = fpath_qa_json_val = config.LOCAL_FILES.FPATH_QA_JSON_VAL
        assert os_path_exists(fpath_qa_json_val), f'dirpath_video does not exist: {fpath_qa_json_val}'
        self.fpath_qa_json_test = fpath_qa_json_test = config.LOCAL_FILES.FPATH_QA_JSON_TEST
        assert os_path_exists(fpath_qa_json_test), f'dirpath_video does not exist: {fpath_qa_json_test}'
        self.fpath_qa_json_merged_to_save = fpath_qa_json_merged_to_save = config.LOCAL_FILES.FPATH_QA_JSON_MERGED_TO_SAVE
        dirpath_qa_json_merged_to_save = os_path_dirname(fpath_qa_json_merged_to_save)
        assert os_access(dirpath_qa_json_merged_to_save, os_W_OK) and os_path_isdir(dirpath_qa_json_merged_to_save), f'dirpath_qa_json_merged_to_save is not writable: {dirpath_qa_json_merged_to_save}'

        # 1. Set up Azure Blob Storage
        self.azure_blob_storage_account_name = config.AZURE_BLOB_STORAGE.ACCOUNT_NAME
        self.azure_blob_storage_key = config.AZURE_BLOB_STORAGE.KEY
        self.azure_blob_storage_container_name = config.AZURE_BLOB_STORAGE.CONTAINER_NAME
        self.azure_blob_storage_sas_token = config.AZURE_BLOB_STORAGE.SAS_TOKEN
        self.container_client = self._get_azure_blob_storage_container_client()

        # 2. Set up Azure Computer Vision
        self.azure_computer_vision_endpoint = config.AZURE_COMPUTER_VISION.ENDPOINT_DOMAIN
        self.azure_computer_vision_key = config.AZURE_COMPUTER_VISION.KEY

        # 1. Set up Azure OpenAI
        self.azure_openai_endpoint = config.AZURE_OPENAI.ENDPOINT_DOMAIN
        self.azure_openai_key = config.AZURE_OPENAI.KEY
        self.azure_openai_deployment = config.AZURE_OPENAI.DEPLOYMENT

        self.azure_openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI.KEY,
            api_version="2023-12-01-preview",
            azure_endpoint=f"https://{self.azure_openai_endpoint}/openai/deployments/gpt4v/extensions/chat/completions"
        )

    def _get_azure_blob_storage_container_client(self):
        azure_blob_storage_container_name = self.azure_blob_storage_container_name
        account_url = f'https://{self.azure_blob_storage_account_name}.blob.core.windows.net'
        blob_service_client = BlobServiceClient(account_url=account_url, credential=self.azure_blob_storage_sas_token)

        azure_blob_storage_container_names = {container_object['name'] for container_object in blob_service_client.list_containers()}

        if not azure_blob_storage_container_names or azure_blob_storage_container_name not in azure_blob_storage_container_names:
            container_client = blob_service_client.create_container(azure_blob_storage_container_name)
        else:
            container_client = blob_service_client.get_container_client(azure_blob_storage_container_name)

        return container_client

    def get_video_urls(self) -> dict:
        '''
        _summary_

        Raises:
            OSError: No videos found with glob pattern {glob_pattern}

        Returns:
            dict:
        '''
        glob_pattern = os_path_join(self.dirpath_videos, '*.mp4')
        fpaths_videos = glob_glob(glob_pattern)
        if not fpaths_videos:
            raise OSError(f'No videos found with glob pattern {glob_pattern}')

        # 1. Check existing videos
        container_client = self.container_client
        fnames_to_upload = {os_path_basename(fpath_video) for fpath_video in fpaths_videos}
        # blobs_names_set = {Path(blob_existing_dict['name']).stem for blob_existing_dict in self.container_client.list_blobs()}
        blobs_names_set = {blob_existing_dict['name'] for blob_existing_dict in container_client.list_blobs()}
        fnames_to_upload = fnames_to_upload.union(blobs_names_set)

        # 2. Upload videos# 1: use Azure Blob Storage client.
        already_exists: bool = blobs_names_set == fnames_to_upload

        dict_video_name_url = {}
        dirpath_videos = self.dirpath_videos

        for fname_to_upload in tqdm(fnames_to_upload):
            with open(os_path_join(dirpath_videos, fname_to_upload), mode='rb') as data:
                if already_exists:
                    blob_client = container_client.get_blob_client(blob=fname_to_upload)
                else:
                    blob_client = container_client.upload_blob(name=fname_to_upload, data=data, overwrite=False)

                blob_name = blob_client.blob_name
                dict_video_name_url[Path(blob_name).stem] = blob_client.url

        blobs_existing_dicts = list(container_client.list_blobs())
        blobs_names_set = {blob_existing_dict['name'] for blob_existing_dict in blobs_existing_dicts}
        assert blobs_names_set == fnames_to_upload

        return dict_video_name_url

    def create_video_index(self, acv_document_id: str):
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{acv_document_id.replace('_', 'underscore')}'
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        data = {'features': [{
                'name': 'vision',
                'domain': 'surveillance'
            }, {
                'name': 'speech',
            }
            ]}
        response = requests_put(url, params=params, headers=headers, json=data, timeout=TIMEOUT)
        response_json = response.json()
        if 'error' in response_json:
            try:
                assert response_json['error']['code'] == 'AlreadyExists'
            except:
                raise SystemExit(f'Failed to create index. Error: {response_json}') from None
            print(f'Index {acv_document_id} already exists')
        return response_json

    def delete_ingestion_index(self, acv_document_id) -> None:
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{acv_document_id.replace('_', 'underscore')}'
        params = {'api-version': '2023-05-01-preview'}
        headers = {'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key}
        response = requests_delete(url, params=params, headers=headers, timeout=TIMEOUT)
        assert response.status_code == 204

    def delete_all_ingestion_index(self) -> None:
        indices = self.get_video_indices()
        return [self.delete_ingestion_index(index) for index in indices]

    def ingest_video(self, video_name: str, video_url: dict):
        acv_document_id = f'i{Path(video_name).stem}'
        # If the index doesn't exist, the problem is likely that the video has never been ingested.
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        video = {
            'mode': 'add',  # Must correspond to acv_document_id exactly.
            'documentId': acv_document_id,
            'documentUrl': video_url,
        }
        data = {'videos': [video], 'includeSpeechTranscript': True}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{acv_document_id.replace('_', 'underscore')}/ingestions/{acv_document_id}'
        try:
            response_batch = requests_put(url, params=params, headers=headers, json=data, timeout=TIMEOUT)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request={data}. Error: {e}') from e

        try:
            response_batch.raise_for_status()
        except Exception as e:
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        return response_batch, acv_document_id

    def get_ingestion_status(self, acv_document_id: str):
        '''
        _summary_

        Raises:
            SystemExit: _description_

        Returns:
            dict: JSON representation of the status of all ingestions for the index
        '''
        assert acv_document_id.startswith('i') # and not acv_document_id.startswith('ii')
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{acv_document_id.replace('_', 'underscore')}/ingestions/{acv_document_id}'
        try:
            response = httpx_get(url, params=params, headers=headers, timeout=TIMEOUT)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request. Error: {e}') from e

        try:
            response.raise_for_status()
        except Exception as e:
            print(response.request.text)
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        response_json = response.json()
        state = response_json['state']
        return state

    def get_video_indices(self) -> set[str]:
        '''
        _summary_

        Raises:
            SystemExit: _description_

        Returns:
            set[str]: {
                        '0k3gj5ybo67',
                        '14yz2qkzve1',
                        '3q9c1zkwm3u',
                        '5t70tei07fa',
                        '68n38j1dxty',
                        '7pbbj26od1b',
                        '857dn4t913p',
                        '9cjfic15jdu',
                        'bn14yjdtf3j',
                        'c8zigj16a16'
                      }
        '''
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes'
        try:
            response = httpx_get(url, params=params, headers=headers, timeout=TIMEOUT)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request. Error: {e}') from e

        try:
            response.raise_for_status()
        except Exception as e:
            print(response.request.text)
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        response_json = response.json()
        return {index_dict['name'] for index_dict in response_json['value']}

    def _ask_one_question(self, acv_document_id: str, question: str, url_video: str, timeout: int) -> dict:
        '''
        _summary_

        Args:
            question (str): _description_
            url_video (str): _description_

        Raises:
            SystemExit: _description_

        Returns:
            dict: _description_
        '''
        assert acv_document_id.startswith('i')

        url_gpt4v_endpoint = f"https://{self.azure_openai_endpoint}/openai/deployments/gpt4v/extensions/chat/completions"
        api_version = '2023-07-01-preview'
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_openai_key,
            "chatgpt_url": f'{url_gpt4v_endpoint}?api-version={api_version}',
        }
        params = {'api-version': api_version}
        roleInformation = "Please analyze the visual and audio information in the uploaded video and output the relevance of each video frame to a given question requiring theory-of-mind reasoning of the video. The output format needs to be a Pythoon dictionary of timestamp: relevance score (a floating value between 0.0 and 1.0, 1.0 meaning highly relevant), each line followed by a Python comment containing the rationale behind the score."
        json_data = {
            'dataSources': [{
                'type': 'AzureComputerVisionVideoIndex',
                'parameters': {
                    'computerVisionBaseUrl': f'https://{self.azure_computer_vision_endpoint}/computervision',
                    'computerVisionApiKey': self.azure_computer_vision_key,
                    'indexName': acv_document_id.replace('_', 'underscore'),
                    'videoUrls': [url_video],
                    'roleInformation': roleInformation,
                },
            }],
            'enhancements': {'video': {'enabled': True}},
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                                "type": "text",
                                "text": roleInformation
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                                "type": "text",
                                "text": "Suppose you are given a 60-second video where a woman is throwing out her mother's food. How relevant is each video frame to answering the question: \"How does the older woman feel when the woman in white starts throwing out her food?\"? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question. How relevant is each video frame to answering the question: \"How does the older woman feel when the woman in white starts throwing out her food?\"? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "```python\n{\n    \"00:00:06.0050000\": 0.1,  # Older woman not in focus, no interaction shown.\n    \"00:00:15.0130000\": 0.2,  # Older woman is present but no interaction yet.\n    \"00:00:16.0130000\": 0.2,  # Still no interaction, relevance is low.\n    \"00:00:17.0140000\": 0.2,  # Older woman in frame but not reacting.\n    \"00:00:18.0150000\": 0.2,  # No clear reaction from the older woman.\n    \"00:00:19.0160000\": 0.2,  # Older woman visible, no interaction observed.\n    \"00:00:20.0170000\": 0.2,  # Older woman in frame, still no reaction.\n    \"00:00:22.0180000\": 0.2,  # Older woman in view, no visible reaction.\n    \"00:00:24.0200000\": 0.3,  # Older woman is visible, potential lead-up to interaction.\n    \"00:00:30.0250000\": 0.4,  # Woman in white approaches fridge, older woman's reaction may start soon.\n    \"00:00:31.0260000\": 0.5,  # Older woman is in focus, beginning of interaction.\n    \"00:00:37.0310000\": 0.7,  # Older woman starts to react, more relevant to the question.\n    \"00:00:38.0320000\": 0.8,  # Older woman's body language starts showing her feelings.\n    \"00:00:39.0330000\": 0.9,  # Older woman's body language clearly shows her feelings.\n    \"00:00:40.0330000\": 1.0,  # Older woman's reaction is fully visible and directly relevant to the question.\n    \"00:00:41.0340000\": 0.9,  # Continuation of the older woman's reaction.\n    \"00:00:43.0360000\": 0.8,  # Older woman's reaction is still ongoing, slightly less relevant as it's not a new response.\n    \"00:00:48.0400000\": 0.7,  # Older woman's reaction is winding down, less relevant.\n    \"00:00:49.0410000\": 0.6,  # Interaction concluding, relevance to feelings decreasing.\n    \"00:00:50\": 0.5,  # Interaction is over, relevance to the question is moderate.\n}\n```"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "acv_document_id",
                            "acv_document_id": acv_document_id,
                        },
                        {
                            "type": "text",
                            "text": f'{PROMPT_BEFORE}"{question}"{PROMPT_AFTER}',
                        }
                    ]
                },
            ],
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 800,
            'stream': False,
        }
        try:
            response = requests_post(url_gpt4v_endpoint, params=params, headers=headers, json=json_data, timeout=timeout)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request for acv_document_id={acv_document_id}. Error: {e}') from e
        return response

    def ask_questions_one_video(self, video_id: str, url_video: str, questions: list[dict], timeout: int, sleep_seconds: int, max_retries, pbar_video, failed_ingestions, delete_and_recreate_index: bool) -> tuple[list[dict], list[dict]]:
        '''
        Load questions from file

        Args:
            dict_video_name_url (dict):

        Raises:
            AssertionError: _description_

        Returns:
            dict: _description_
        '''
        acv_document_id = f'i{video_id}'
        # Case 1: the video has never been ingested. The index doesn't exist
        if acv_document_id not in self.get_video_indices():
            self.create_video_index(acv_document_id)
            self.ingest_video_with_retries(video_id, url_video, max_retries, pbar_video, failed_ingestions, delete_and_recreate_index=False)

        successes, failures, requests = [], [], []
        _ask_one_question = self._ask_one_question
        for question in (pbar_question := tqdm(questions, leave=False)):
            pbar_question.set_description(f"ask_questions: Processing question {question['q']}")
            try:
                response = _ask_one_question(acv_document_id, question['q'], url_video, timeout)
            except RequestException:
                failures.append((acv_document_id, question))
                requests.append((acv_document_id, question))
                continue
            if response.status_code == 500:
                failures.append((acv_document_id, question))
                requests.append((acv_document_id, question))
                continue
            # Case 2: the video has been successfully ingested but it's not available. In this case, recreate the index and reingest the video.
            if 'document does not exist' in response.text.lower():
                pbar_video.write(f'document does not exist for video={video_id}. Ingesting')
                self.ingest_video_with_retries(video_id, url_video, max_retries, pbar_video, failed_ingestions, delete_and_recreate_index=True)
                return self.ask_questions_one_video(video_id, url_video, questions, timeout, sleep_seconds, max_retries, pbar_video, failed_ingestions, delete_and_recreate_index)
            else:
                successes.append(response)
            requests.append(response.request)

            # count += 1
            sleep(sleep_seconds)
        return successes, failures, requests

    def ingest_video_with_retries(self, video_id: str, url_video: str, max_retries: int, pbar_video, failed_ingestions: list, delete_and_recreate_index: bool):
        ingest_video = self.ingest_video
        get_ingestion_status = self.get_ingestion_status
        acv_document_id = f'i{video_id}'
        if delete_and_recreate_index:
            pbar_video.write(f"Deleting and recreating index for video {video_id}")
            self.delete_ingestion_index(acv_document_id)
            sleep(10)
        response, acv_document_id = ingest_video(video_id, url_video)
        ingestion_status_video = get_ingestion_status(acv_document_id)
        trys = 0
        while ingestion_status_video != 'Completed':
            if trys >= max_retries:
                pbar_video.write(f"Exceeded max_retries={max_retries} while ingesting video {acv_document_id}. Got status={ingestion_status_video}")
                failed_ingestions.append((acv_document_id, ingestion_status_video, response))
                break
            sleep(1)
            ingestion_status_video = get_ingestion_status(acv_document_id)
            if ingestion_status_video == 'Running':
                trys += 0.1
            else:
                trys += 1

    def ask_questions_all_videos(self, dict_video_name_url: dict, sleep_seconds: int=10, timeout: int=TIMEOUT, videos_max: int=-1) -> list[dict]:
        '''
        Load questions from file

        Args:
            dict_video_name_url (dict):

        Raises:
            AssertionError: _description_

        Returns:
            dict: _description_
        '''
        # 1. Load the questions from file
        dict_video_qa_merged = self.merge_and_save_multiple_dicts()
        video_keys_existing = dict_video_qa_merged.keys() & dict_video_name_url.keys()
        ask_questions_one_video = self.ask_questions_one_video
        result_dict = {}
        len_video_keys_existing = len(video_keys_existing)
        total = len_video_keys_existing if videos_max < 0 else min(len_video_keys_existing, videos_max)
        dirname_result = 'result'
        max_retries = 10
        failed_ingestions = []

        video_keys_incomplete = {video_key_existing for video_key_existing in video_keys_existing if not os_path_exists(os_path_join(dirname_result, f'{video_key_existing}.pkl'))}
        print(f'There are {len(video_keys_incomplete)} incomplete keys out of {len(video_keys_existing)}')

        for i, video_id in (pbar_video := tqdm(enumerate(video_keys_existing), total=min(total, len(video_keys_incomplete)))):
            if i >= total:
                break
            if os_path_exists(os_path_join(dirname_result, f'{video_id}.pkl')):
                continue
            # if video_id not in dict_final_dict: # NOTE: toggled this logic.
                # continue
            pbar_video.set_description(f"ask_questions: Processing video {i}/{total}: {video_id}")
            questions = dict_video_qa_merged[video_id]
            url_video = dict_video_name_url[video_id]

            questions = [questions[0]] + questions
            successes, failures, requests = ask_questions_one_video(video_id, url_video, questions, timeout, sleep_seconds, max_retries, pbar_video, failed_ingestions, delete_and_recreate_index=False)
            dict_current_video = {'video_id': video_id, 'successes': successes, 'failures': failures, 'requests': requests}
            result_dict[video_id] = dict_current_video
            if len(successes) == 0:
                video_id += '_failed'
            with open(f'./result/{video_id}.pkl', 'wb') as f:
                pickle_dump(dict_current_video, f)

        result_dict['failed_ingestions'] = failed_ingestions
        return result_dict


    def merge_and_save_multiple_dicts(self) -> dict:
        '''
        _summary_

        Returns:
            dict: _description_
        '''
        dict_vqa_train = _load_vqa_file(self.fpath_qa_json_train)
        dict_vqa_val = _load_vqa_file(self.fpath_qa_json_val)
        dict_vqa_test = _load_vqa_file(self.fpath_qa_json_test)
        dict_vqa_merged = {**dict_vqa_train, **dict_vqa_val, **dict_vqa_test}
        with open(self.fpath_qa_json_merged_to_save, 'w') as file_out:
            json_dump(dict_vqa_merged, file_out, indent=4)
        return dict_vqa_merged
