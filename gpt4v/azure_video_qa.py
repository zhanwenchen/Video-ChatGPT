
from os import access as os_access, W_OK as os_W_OK
from os.path import exists as os_path_exists, join as os_path_join, basename as os_path_basename, dirname as os_path_dirname, isdir as os_path_isdir
from glob import glob as glob_glob
from json import load as json_load, dump as json_dump
from json.decoder import JSONDecodeError
from collections import defaultdict
from pathlib import Path
from time import sleep
# from urllib.parse import unquote
from pickle import dump as pickle_dump, load as pickle_load
from httpx import get as httpx_get
from requests import post as requests_post, put as requests_put, delete as requests_delete, RequestException
from tqdm import tqdm
from tqdm.notebook import tqdm  # pip install ipywidgets
# from tqdm.contrib import tenumerate
from azure.storage.blob import BlobServiceClient
# from config import get_cfg_defaults
# from curlify import to_curl
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
        self.azure_computer_vision_index_name = config.AZURE_COMPUTER_VISION.VIDEO_INDEX_NAME

        # 1. Set up Azure OpenAI
        self.azure_openai_endpoint = config.AZURE_OPENAI.ENDPOINT_DOMAIN
        self.azure_openai_key = config.AZURE_OPENAI.KEY
        self.azure_openai_deployment = config.AZURE_OPENAI.DEPLOYMENT

        self.azure_openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI.KEY,
            api_version="2023-12-01-preview",
            azure_endpoint=f"https://{self.azure_openai_endpoint}/openai/deployments/gpt4v/extensions/chat/completions"
        )

        # deployment_name='REPLACE_WITH_YOUR_DEPLOYMENT_NAME' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment.

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

    def create_video_index(self):
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}'
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
            assert response_json['error']['code'] == 'AlreadyExists'
            print(f'Index {self.azure_computer_vision_index_name} already exists')
        return response_json

    def delete_ingestion_index(self) -> None:
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}'
        params = {'api-version': '2023-05-01-preview'}
        headers = {'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key}
        response = requests_delete(url, params=params, headers=headers, timeout=TIMEOUT)
        assert response.status_code == 204

    def ingest_video(self, video_name: str, video_url: dict):
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        acv_document_id = f'i{Path(video_name).stem}'
        video = {
            'mode': 'add',  # Must correspond to acv_document_id exactly.
            'documentId': acv_document_id,
            'documentUrl': video_url,
        }
        data = {'videos': [video], 'includeSpeechTranscript': True}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}/ingestions/{acv_document_id}'
        try:
            response_batch = requests_put(url, params=params, headers=headers, json=data, timeout=TIMEOUT)
        except RequestException as e:
            # print(f'request = {to_curl(response_batch.request, pretty=True)}')
            raise SystemExit(f'Failed to make the request={data}. Error: {e}') from e

        # print(response_batch.text)
        try:
            response_batch.raise_for_status()
        except Exception as e:
            # print(response_batch.request.text)
            # print(f'request = {to_curl(response_batch.request, pretty=True)}')
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        # responses_jsons.append(response_batch.json())
        # sleep(0.5)
        return response_batch, acv_document_id

    def delete_ingest_video(self, video_name: str, video_url: dict):
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        acv_document_id = f'i{Path(video_name).stem}'
        video = {
            'mode': 'add',  # Must correspond to acv_document_id exactly.
            'documentId': acv_document_id,
            'documentUrl': video_url,
        }
        data = {'videos': [video]}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}/ingestions/{acv_document_id}'
        try:
            response_batch = requests_delete(url, params=params, headers=headers, json=data, timeout=TIMEOUT)
        except RequestException as e:
            # print(f'request = {to_curl(response_batch.request, pretty=True)}')
            raise SystemExit(f'Failed to make the request. Error: {e}') from e

        # print(response_batch.text)
        try:
            response_batch.raise_for_status()
        except Exception as e:
            # print(response_batch.request.text)
            # print(f'request = {to_curl(response_batch.request, pretty=True)}')
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        # responses_jsons.append(response_batch.json())
        # sleep(0.5)
        return response_batch, acv_document_id

    def ingest_videos(self, dict_video_name_url: dict) -> list[dict]:
        videos = [{
            'mode': 'add',
            'documentId': f'i{Path(video_name).stem}', # Must correspond to acv_document_id exactly.
            'documentUrl': video_url,
        } for video_name, video_url in dict_video_name_url.items()]
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        responses_jsons = []
        azure_computer_vision_endpoint = self.azure_computer_vision_endpoint
        azure_computer_vision_index_name = self.azure_computer_vision_index_name
        # for i, videos_batch in tenumerate(batched(videos, batch_size)):
        for video in (pbar := tqdm(videos)):
            data = {'videos': [video]}
            acv_document_id = video['documentId']
            pbar.set_description(f"ingest_videos: Processing {acv_document_id}")
            url = f'https://{azure_computer_vision_endpoint}/computervision/retrieval/indexes/{azure_computer_vision_index_name}/ingestions/{acv_document_id}'
            try:
                response_batch = requests_put(url, params=params, headers=headers, json=data, timeout=TIMEOUT)
            except RequestException as e:
                # print(f'request = {to_curl(response_batch.request, pretty=True)}')
                raise SystemExit(f'Failed to make the request. Error: {e}') from e

            # print(response_batch.text)
            try:
                response_batch.raise_for_status()
            except Exception as e:
                # print(response_batch.request.text)
                # print(f'request = {to_curl(response_batch.request, pretty=True)}')
                raise SystemExit(f'Failed to make the request. Error: {e}') from e
            responses_jsons.append(response_batch.json())
            sleep(0.5)
        return responses_jsons

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
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}/ingestions/{acv_document_id}'
        try:
            response = httpx_get(url, params=params, headers=headers, timeout=TIMEOUT)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request. Error: {e}') from e

        # print(response.text)
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.request.text)
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        response_json = response.json()
        state = response_json['state']
        return state

    def get_ingestion_statuses(self) -> dict:
        '''
        _summary_

        Raises:
            SystemExit: _description_

        Returns:
            dict: JSON representation of the status of all ingestions for the index
        '''
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_computer_vision_key,
            'Content-Type': 'application/json',
        }
        params = {'api-version': '2023-05-01-preview'}
        url = f'https://{self.azure_computer_vision_endpoint}/computervision/retrieval/indexes/{self.azure_computer_vision_index_name}/ingestions'
        try:
            response = httpx_get(url, params=params, headers=headers, timeout=TIMEOUT)
        except RequestException as e:
            raise SystemExit(f'Failed to make the request. Error: {e}') from e

        print(response.text)
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.request.text)
            raise SystemExit(f'Failed to make the request. Error: {e}') from e
        response_json = response.json()
        return response_json

    def _ask_one_question_new(self, acv_document_id: str, question: str, url_video: str, timeout: int):

        # Send a completion call to generate an answer
        print('Sending a test completion job')
        # start_phrase = 'Write a tagline for an ice cream shop. '
        # response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=10)
        # print(start_phrase+response.choices[0].text)
        response = self.azure_openai_client.chat.completions.create(
            model=self.azure_openai_deployment,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [
                    {
                        "type": "acv_document_id",
                        "acv_document_id": acv_document_id,
                    },
                    {
                        "type": "text",
                        "text": "Describe this video:"
                    }
                ] }
            ],
            extra_body={
                "dataSources": [
                    {
                        "type": "AzureComputerVisionVideoIndex",
                        "parameters": {
                            "computerVisionBaseUrl": f'https://{self.azure_computer_vision_endpoint}/computervision', # your endpoint should look like the following https://YOUR_RESOURCE_NAME.cognitiveservices.azure.com/computervision
                            "computerVisionApiKey": self.azure_computer_vision_key,
                            "indexName": self.azure_computer_vision_index_name,
                            "videoUrls": [url_video],
                        }
                    }],
                "enhancements": {
                    "video": {
                        "enabled": True
                    }
                }
            },
            max_tokens=100
        )
        return response

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
        assert acv_document_id.startswith('i') # and not acv_document_id.startswith('ii')

        url_gpt4v_endpoint = f"https://{self.azure_openai_endpoint}/openai/deployments/gpt4v/extensions/chat/completions"
        api_version = '2023-07-01-preview'
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_openai_key,
            "chatgpt_url": f'{url_gpt4v_endpoint}?api-version={api_version}',
            # 'accept': 'application/json',
        }
        params = {'api-version': api_version}
        roleInformation = 'You are an AI assistant that helps people find information related to videos and theory of mind.'
        json_data = {
            'dataSources': [{
                'type': 'AzureComputerVisionVideoIndex',
                'parameters': {
                    'computerVisionBaseUrl': f'https://{self.azure_computer_vision_endpoint}/computervision',
                    'computerVisionApiKey': self.azure_computer_vision_key,
                    'indexName': self.azure_computer_vision_index_name,
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
                            "text": "Please analyze the visual and audio information in the uploaded video and output the relevance of each video frame to a given question requiring theory-of-mind reasoning of the video. The output format needs to be a Pythoon dictionary of timestamp: relevance score (a floating value between 0.0 and 1.0, 1.0 meaning highly relevant), each line followed by a Python comment containing the rationale behind the score."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How relevant is each video frame to answering the question: \"How does the older woman feel when the woman in white starts throwing out her food?\"? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "{  \n  \"00:00:06.005\": 0.0, # No interaction related to throwing out food yet.  \n  \"00:00:15.013\": 0.2, # Older woman is present, potential buildup to the event.  \n  \"00:00:16.013\": 0.2, # Continuation of the previous frame, still no clear reaction.  \n  \"00:00:17.014\": 0.2, # Similar to previous frames, no change in situation.  \n  \"00:00:18.015\": 0.2, # Same as above, no visible reaction to the event.  \n  \"00:00:19.016\": 0.2, # No visible change in the situation or reaction.  \n  \"00:00:20.017\": 0.2, # Still no action regarding the food throwing.  \n  \"00:00:22.018\": 0.2, # Scene continues without change in the older woman’s status.  \n  \"00:00:24.020\": 0.5, # The woman in white is near the fridge, indicating the event might start soon.  \n  \"00:00:30.025\": 0.5, # The woman in white is interacting with the fridge, which may lead to the event.  \n  \"00:00:31.026\": 0.7, # The older woman has a clear reaction, indicating the event has begun.  \n  \"00:00:37.031\": 0.8, # Direct interaction between the two characters, relevant to the event.  \n  \"00:00:38.032\": 1.0, # The older woman’s reaction is visible as the woman in white holds the food.  \n  \"00:00:39.033\": 1.0, # Continuation of the highly relevant interaction.  \n  \"00:00:40.033\": 1.0, # Ongoing relevant interaction depicting the older woman's feelings.  \n  \"00:00:41.034\": 1.0, # Direct confrontation about the food, highly relevant.  \n  \"00:00:43.036\": 1.0, # The interaction continues, showing the older woman's feelings.  \n  \"00:00:48.040\": 0.8, # The event is concluding but still shows the older woman's reaction.  \n  \"00:00:49.041\": 0.8, # The aftermath of the event, still relevant to the older woman's feelings.  \n  \"00:00:50\": 0.7, # The situation is cooling down, but the older woman's feelings might still be deduced.  \n} "
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "acv_document_id",
                            "acv_document_id": "AOAIChatDocument"
                        },
                        {
                            "type": "text",
                            "text": "\nHow relevant is each video frame to answering the question: \"How does the older woman feel when the woman in white starts throwing out her food?\"? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question. How relevant is each video frame to answering the question: \"How does the older woman feel when the woman in white starts throwing out her food?\"? Please output a dictionary of each video frame to its corresponding relevance score between 0.0 and 1.0 to the given question.\n"
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
                'role': 'user',
                'content': [{
                    'type': 'acv_document_id',
                    # 'acv_document_id': f'i{video_id.lower()}',
                    'acv_document_id': acv_document_id,
                }, {
                    'type': 'text',
                    'text': f'{PROMPT_BEFORE}"{question}"{PROMPT_AFTER}',
                }],
            }],
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

    def ask_questions_one_video(self, video_id: str, url_video: str, questions: list[dict], timeout: int, sleep_seconds: int) -> tuple[list[dict], list[dict]]:
        '''
        Load questions from file

        Args:
            dict_video_name_url (dict):

        Raises:
            AssertionError: _description_

        Returns:
            dict: _description_
        '''
        successes, failures, requests = [], [], []
        _ask_one_question = self._ask_one_question
        for question in (pbar_question := tqdm(questions, leave=False)):
            pbar_question.set_description(f"ask_questions: Processing question {question['q']}")
            try:
                response = _ask_one_question(video_id, question['q'], url_video, timeout)
            except RequestException:
                failures.append((video_id, question))
                requests.append((video_id, question))
                continue
            if 'Document does not exist' in response.text:
                failures.append(response)
            else:
                successes.append(response)
            requests.append(response.request)

            # count += 1
            sleep(sleep_seconds)
        # print(f'ask_questions: DONE. video_id={video_id}. len(successes)={len(successes)}. len(failures)={len(failures)}')
        return successes, failures, requests

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
        # _ask_one_question = self._ask_one_question
        # responses = []
        # failed = []
        # num_skipped = 0
        # dict_video_stem_url = {Path(video_name).stem: video_url for video_name, video_url in dict_video_name_url.items()}
        video_keys_existing = dict_video_qa_merged.keys() & dict_video_name_url.keys()
        # count = 0
        ask_questions_one_video = self.ask_questions_one_video
        result_dict = {}
        len_video_keys_existing = len(video_keys_existing)
        total = len_video_keys_existing if videos_max < 0 else min(len_video_keys_existing, videos_max)
        # for video_id in (pbar_video := tqdm(video_keys_existing, total=total)):
        get_ingestion_status = self.get_ingestion_status
        ingest_video = self.ingest_video
        dirname_result = 'result'
        max_retrys = 10
        failed_ingestions = []

        with open('final_dict.pkl', 'rb') as f:
            dict_final_dict = pickle_load(f)

        for i, video_id in (pbar_video := tqdm(enumerate(video_keys_existing), total=total)):
            if i >= total:
                break
            if os_path_exists(os_path_join(dirname_result, f'{video_id}.pkl')):
                continue
            # if video_id not in dict_final_dict: # NOTE: toggled this logic.
                # continue
            pbar_video.set_description(f"ask_questions: Processing video {i}/{total}: {video_id}")
            questions = dict_video_qa_merged[video_id]
            # if video_id not in dict_video_stem_url:
            #     num_skipped += 1
            #     print(f'Skipping video_id: {video_id} because the video is missing. num_skipped={num_skipped}')
            #     continue
            # print(f'Found video_id: {video_id}. num_found={len(responses)+1}')
            url_video = dict_video_name_url[video_id]
            response, acv_document_id = ingest_video(video_id, url_video)
            # pbar_video.write("Ingesting video...")
            # url_video = dict_video_stem_url[video_id]
            # acv_document_id = f'i{video_id}'
            ingestion_status_video = get_ingestion_status(acv_document_id)
            trys = 0
            while ingestion_status_video != 'Completed':
                if trys >= max_retrys:
                    pbar_video.write(f"Problem ingesting video {acv_document_id}. Got status={ingestion_status_video}")
                    failed_ingestions.append((acv_document_id, ingestion_status_video, response))
                    break
                # print(f'{acv_document_id}: ingestion_status_video={ingestion_status_video}')
                sleep(1)
                ingestion_status_video = get_ingestion_status(acv_document_id)
                if ingestion_status_video == 'Running':
                    trys += 0.1
                else:
                    trys += 1
            # pbar_video.write("Done ingesting video")
            # match ingestion_status_video:
            #     case 'Completed':
            #         print(f'{acv_document_id}: Completed')
            #         pass
            #     case 'Running':
            #         sleep(60)
            #     case _:
            #         raise ValueError(f'ingestion_status_video={ingestion_status_video}')
            # sleep(2)
            questions = [questions[0]] + questions
            successes, failures, requests = ask_questions_one_video(acv_document_id, url_video, questions, timeout, sleep_seconds)
            dict_current_video = {'video_id': video_id, 'successes': successes, 'failures': failures, 'requests': requests}
            result_dict[video_id] = dict_current_video
            # for question in (pbar_question := tqdm(questions, leave=False)):
            #     pbar_question.set_description(f"ask_questions: Processing question {question['q']}")

            #     if count >= questions_max and questions_max > 0:
            #         print(f'ask_questions: DONE. len(responses)={len(responses)}. len(responses)={len(failed)}')
            #         return responses, failed
            #     try:
            #         response = _ask_one_question(video_id, question['q'], url_video, timeout)
            #     except RequestException:
            #         failed.append((video_id, question))

            #     if 'Document does not exist' in response.text:
            #         failed.append(response)
            #     else:
            #         responses.append(response)

            #     count += 1
            #     # failed.append(response)
            #     # breakpoint()
            #     sleep(sleep_seconds)
            #     # responses.append(response)
            # sleep(sleep_seconds)
            with open(f'./result/{video_id}.pkl', 'wb') as f:
                pickle_dump(dict_current_video, f)

        # print(f'ask_questions: DONE. len(responses)={len(responses)}. len(responses)={len(failed)}')
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

# def main():
#     cfg = get_cfg_defaults()
#     cfg.freeze()
#     print(cfg)

#     azure_qa = AzureVideoQA(cfg)


# if __name__ == '__main__':
#     main()


# data: {"choices":[{"messages":[{"delta":{"role":"tool", "content": "Retrieving frames from your video"}}]}]}

# data: {"choices":[{"messages":[{"delta":{"role":"tool", "content": "Found following frames from your video: { 00:00:47.0390000, }"}}]}]}

# ﻿data: {"id":"","object":"","created":0,"model":"","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}],"choices":[]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{},"messages":[{"delta":{"role":"assistant","content":null}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":"I"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":"'m"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" sorry"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":","}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" I"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" cannot"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" assist"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" with"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" that"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":" request"}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":null,"index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"messages":[{"delta":{"content":"."}}]}]}

# data: {"id":"chatcmpl-8zf64GnBNLPlvAtoo0jr0RD4uy4pB","object":"chat.completion.chunk","created":1709706676,"model":"gpt-4","choices":[{"finish_reason":"stop","index":0,"content_filter_results":{},"messages":[{"delta":{"content":null}}]}]}

# data: [DONE]
