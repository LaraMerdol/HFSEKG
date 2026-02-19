import concurrent.futures
import json
import logging
import os
from datetime import datetime
from neo4j import GraphDatabase
import requests
from huggingface_hub import (get_dataset_tags, get_model_tags, list_datasets,
                             list_models, list_spaces, login, logout)
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry


class KGConstructor:
    def __init__(self):
        self.setup_environment()
        self.init_data_structures()
        
        self.setup_session()

    def setup_neo4j_connection(self):
        """Setup connection to local Neo4j instance"""
        try:
            self.neo4j_uri = "bolt://localhost:7687"
            self.neo4j_user = "neo4j"
            self.neo4j_password = "01234567"  # Change this to your Neo4j password
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logging.info("[neo4j] Successfully connected to Neo4j at " + self.neo4j_uri)
        except Exception as e:
            logging.error(f"[neo4j] Failed to connect to Neo4j: {e}")
            self.driver = None

    def setup_environment(self):
        """Setup environment variables and logging"""
        self.my_hf_token = "" # Add your Hugging Face API token here
        login(token=self.my_hf_token)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = "HuggingKG_V"+timestamp
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'logs.log'), 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_session(self):
        """Setup requests session with retry mechanism"""
        self.session = requests.Session()
        retries = Retry(
            total=10, 
            backoff_factor=1, 
            status_forcelist=[408, 429, 502, 503, 504, 522, 524],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
            respect_retry_after_header=True
            )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=200,
            pool_maxsize=200,
            pool_block=True
            )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Authorization': f"Bearer {self.my_hf_token if self.my_hf_token else ''}"
        })

    def init_data_structures(self):
        # entity data
        self.processed_tasks = []
        self.processed_models = []
        self.processed_datasets = []
        self.processed_spaces = []
        self.processed_papers = []
        self.processed_collections = []
        self.processed_users = []
        self.processed_orgs = []

        # relation data
        self.model_definedFor_task = []
        self.model_adapter_model = []
        self.model_finetune_model = []
        self.model_merge_model = []
        self.model_quantized_model = []
        self.model_trainedOrFineTunedOn_dataset = []
        self.model_cite_paper = []
        self.dataset_definedFor_task = []
        self.dataset_cite_paper = []
        self.space_use_model = []
        self.space_use_dataset = []
        self.collection_contain_model = []
        self.collection_contain_dataset = []
        self.collection_contain_space = []
        self.collection_contain_paper = []
        self.username_publish_model = []
        self.username_publish_dataset = []
        self.username_publish_space = []
        self.user_publish_model = []
        self.user_publish_dataset = []
        self.user_publish_space = []
        self.user_publish_paper = []
        self.user_own_collection = []
        self.user_like_model = []
        self.user_like_dataset = []
        self.user_like_space = []
        self.user_follow_user = []
        self.user_affiliatedWith_org = []
        self.user_follow_org = []
        self.org_publish_model = []
        self.org_publish_dataset = []
        self.org_publish_space = []
        self.org_own_collection = []

        # extra data
        self.task_ids = set()
        self.model_ids = set()
        self.dataset_ids = set()
        self.space_ids = set()
        self.paper_ids = set()
        self.collection_slugs = set()
        self.user_ids = set()
        self.org_ids = set()

        self.arxiv_ids = set()
        self.username_ids = set()
        # self.visited_user_ids = set()
        # self.visited_org_ids = set()

    def save_data(self, filename, data):
        """Helper method to save data to JSON file"""
        with open(os.path.join(self.output_dir, f'{filename}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save_entity_data(self):
        print("Saving entity data...")
        save_entities = [
            ('tasks', self.processed_tasks),
            ('models', self.processed_models),
            ('datasets', self.processed_datasets),
            ('spaces', self.processed_spaces),
            ('papers', self.processed_papers),
            ('collections', self.processed_collections),
            ('users', self.processed_users),
            ('orgs', self.processed_orgs)
        ]
        for entity_name, data in save_entities:
            self.save_data(entity_name, data)

    def save_relation_data(self):
        print("Saving relation data...")
        save_relations = [
            ('model_definedFor_task', self.model_definedFor_task),
            ('model_adapter_model', self.model_adapter_model),
            ('model_finetune_model', self.model_finetune_model),
            ('model_merge_model', self.model_merge_model),
            ('model_quantized_model', self.model_quantized_model),
            ('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset),
            ('model_cite_paper', self.model_cite_paper),
            ('dataset_definedFor_task', self.dataset_definedFor_task),
            ('dataset_cite_paper', self.dataset_cite_paper),
            ('space_use_model', self.space_use_model),
            ('space_use_dataset', self.space_use_dataset),
            ('collection_contain_model', self.collection_contain_model),
            ('collection_contain_dataset', self.collection_contain_dataset),
            ('collection_contain_space', self.collection_contain_space),
            ('collection_contain_paper', self.collection_contain_paper),
            ('user_publish_model', self.user_publish_model),
            ('user_publish_dataset', self.user_publish_dataset),
            ('user_publish_space', self.user_publish_space),
            ('user_publish_paper', self.user_publish_paper),
            ('user_own_collection', self.user_own_collection),
            ('user_like_model', self.user_like_model),
            ('user_like_dataset', self.user_like_dataset),
            ('user_like_space', self.user_like_space),
            ('user_follow_user', self.user_follow_user),
            ('user_affiliatedWith_org', self.user_affiliatedWith_org),
            ('user_follow_org', self.user_follow_org),
            ('org_publish_model', self.org_publish_model),
            ('org_publish_dataset', self.org_publish_dataset),
            ('org_publish_space', self.org_publish_space),
            ('org_own_collection', self.org_own_collection)
        ]
        for relation_name, data in save_relations:
            self.save_data(relation_name, data)

    def save_extra_data(self):
        print("Saving extra data...")
        save_extra_data = [
            # 'extra_data_name', data
            ('task_ids', self.task_ids),
            ('model_ids', self.model_ids),
            ('dataset_ids', self.dataset_ids),
            ('space_ids', self.space_ids),
            ('paper_ids', self.paper_ids),
            ('collection_slugs', self.collection_slugs),
            ('user_ids', self.user_ids),
            ('org_ids', self.org_ids),

            ('arxiv_ids', self.arxiv_ids),
            ('username_ids', self.username_ids),
            ('username_publish_model', self.username_publish_model),
            ('username_publish_dataset', self.username_publish_dataset),
            ('username_publish_space', self.username_publish_space)
        ]
        
        os.makedirs(os.path.join(self.output_dir, 'extra_data'), exist_ok=True)

        for extra_data_name, data in save_extra_data:
            with open(os.path.join(self.output_dir, 'extra_data', f'{extra_data_name}.json'), 'w') as f:
                json.dump(data, f, indent=4)

    def save_all_data(self):
        # print("Saving all data...")
        self.save_entity_data()
        self.save_relation_data()

    def log_stats(self):
        """Log statistics about processed data"""

        entity_stats = {
            'tasks': len(self.processed_tasks),
            'models': len(self.processed_models),
            'datasets': len(self.processed_datasets),
            'spaces': len(self.processed_spaces),
            'papers': len(self.processed_papers),
            'collections': len(self.processed_collections),
            'users': len(self.processed_users),
            'orgs': len(self.processed_orgs)
        }

        relation_stats = {
            'model_definedFor_task': len(self.model_definedFor_task),
            'model_adapter_model': len(self.model_adapter_model),
            'model_finetune_model': len(self.model_finetune_model),
            'model_merge_model': len(self.model_merge_model),
            'model_quantized_model': len(self.model_quantized_model),
            'model_trainedOrFineTunedOn_dataset': len(self.model_trainedOrFineTunedOn_dataset),
            'model_cite_paper': len(self.model_cite_paper),
            'dataset_definedFor_task': len(self.dataset_definedFor_task),
            'dataset_cite_paper': len(self.dataset_cite_paper),
            'space_use_model': len(self.space_use_model),
            'space_use_dataset': len(self.space_use_dataset),
            'collection_contain_model': len(self.collection_contain_model),
            'collection_contain_dataset': len(self.collection_contain_dataset),
            'collection_contain_space': len(self.collection_contain_space),
            'collection_contain_paper': len(self.collection_contain_paper),
            'user_publish_model': len(self.user_publish_model),
            'user_publish_dataset': len(self.user_publish_dataset),
            'user_publish_space': len(self.user_publish_space),
            'user_publish_paper': len(self.user_publish_paper),
            'user_own_collection': len(self.user_own_collection),
            'user_like_model': len(self.user_like_model),
            'user_like_dataset': len(self.user_like_dataset),
            'user_like_space': len(self.user_like_space),
            'user_follow_user': len(self.user_follow_user),
            'user_affiliatedWith_org': len(self.user_affiliatedWith_org),
            'user_follow_org': len(self.user_follow_org),
            'org_publish_model': len(self.org_publish_model),
            'org_publish_dataset': len(self.org_publish_dataset),
            'org_publish_space': len(self.org_publish_space),
            'org_own_collection': len(self.org_own_collection)
        }

        summary_stats = {
            'entities': sum(entity_stats.values()),
            'relations': sum(relation_stats.values())
        }

        logging.info("Entity stats:")
        for entity, count in entity_stats.items():
            logging.info(f"{entity}: {count}")
        
        logging.info("Relation stats:")
        for relation, count in relation_stats.items():
            logging.info(f"{relation}: {count}")
        
        logging.info("Summary stats:")
        for stat, count in summary_stats.items():
            logging.info(f"{stat}: {count}")
    
    def verify_relations(self):
        print("Verifying relations...")
        
        self.task_ids = {task['id'] for task in self.processed_tasks}
        self.model_ids = {model['id'] for model in self.processed_models}
        self.dataset_ids = {dataset['id'] for dataset in self.processed_datasets}
        self.space_ids = {space['id'] for space in self.processed_spaces}
        self.paper_ids = {paper['id'] for paper in self.processed_papers}
        self.collection_slugs = {collection['slug'] for collection in self.processed_collections}
        self.user_ids = {user['id'] for user in self.processed_users}
        self.org_ids = {org['id'] for org in self.processed_orgs}
        
        relation_checks = [
            ('model_definedFor_task', self.model_definedFor_task, 'model_id', 'task_id', self.model_ids, self.task_ids),
            ('model_adapter_model', self.model_adapter_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_finetune_model', self.model_finetune_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_merge_model', self.model_merge_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_quantized_model', self.model_quantized_model, 'base_model_id', 'model_id', self.model_ids, self.model_ids),
            ('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset, 'model_id', 'dataset_id', self.model_ids, self.dataset_ids),
            ('model_cite_paper', self.model_cite_paper, 'model_id', 'arxiv_id', self.model_ids, self.paper_ids),

            ('dataset_definedFor_task', self.dataset_definedFor_task, 'dataset_id', 'task_id', self.dataset_ids, self.task_ids),
            ('dataset_cite_paper', self.dataset_cite_paper, 'dataset_id', 'arxiv_id', self.dataset_ids, self.paper_ids),

            ('space_use_model', self.space_use_model, 'space_id', 'model_id', self.space_ids, self.model_ids),
            ('space_use_dataset', self.space_use_dataset, 'space_id', 'dataset_id', self.space_ids, self.dataset_ids),

            ('collection_contain_model', self.collection_contain_model, 'collection_slug', 'model_id', self.collection_slugs, self.model_ids),
            ('collection_contain_dataset', self.collection_contain_dataset, 'collection_slug', 'dataset_id', self.collection_slugs, self.dataset_ids),
            ('collection_contain_space', self.collection_contain_space, 'collection_slug', 'space_id', self.collection_slugs, self.space_ids),
            ('collection_contain_paper', self.collection_contain_paper, 'collection_slug', 'paper_id', self.collection_slugs, self.paper_ids),

            ('user_publish_model', self.user_publish_model, 'user_id', 'model_id', self.user_ids, self.model_ids),
            ('user_publish_dataset', self.user_publish_dataset, 'user_id', 'dataset_id', self.user_ids, self.dataset_ids),
            ('user_publish_space', self.user_publish_space, 'user_id', 'space_id', self.user_ids, self.space_ids),
            ('user_publish_paper', self.user_publish_paper, 'user_id', 'paper_id', self.user_ids, self.paper_ids),
            ('user_own_collection', self.user_own_collection, 'user_id', 'collection_slug', self.user_ids, self.collection_slugs),
            ('user_like_model', self.user_like_model, 'user_id', 'model_id', self.user_ids, self.model_ids),
            ('user_like_dataset', self.user_like_dataset, 'user_id', 'dataset_id', self.user_ids, self.dataset_ids),
            ('user_like_space', self.user_like_space, 'user_id', 'space_id', self.user_ids, self.space_ids),
            ('user_follow_user', self.user_follow_user, 'follower_id', 'followee_id', self.user_ids, self.user_ids),
            ('user_affiliatedWith_org', self.user_affiliatedWith_org, 'user_id', 'org_id', self.user_ids, self.org_ids),
            ('user_follow_org', self.user_follow_org, 'user_id', 'org_id', self.user_ids, self.org_ids),

            ('org_publish_model', self.org_publish_model, 'org_id', 'model_id', self.org_ids, self.model_ids),
            ('org_publish_dataset', self.org_publish_dataset, 'org_id', 'dataset_id', self.org_ids, self.dataset_ids),
            ('org_publish_space', self.org_publish_space, 'org_id', 'space_id', self.org_ids, self.space_ids),
            ('org_own_collection', self.org_own_collection, 'org_id', 'collection_slug', self.org_ids, self.collection_slugs)
        ]
    
        invalid_relation_count = 0

        for relation_name, relations, id1_key, id2_key, valid_ids1, valid_ids2 in relation_checks:
            valid_relations = []
            for relation in relations:
                if relation[id1_key] in valid_ids1 and relation[id2_key] in valid_ids2:
                    valid_relations.append(relation)
                else:
                    logging.warning(f"[constructor] Invalid relation found in {relation_name}: {relation}")
                    invalid_relation_count += 1
            relations[:] = valid_relations

        if invalid_relation_count > 0:
            logging.warning(f"[constructor] Found {invalid_relation_count} invalid relations")

        # save data after removing invalid relations
        # print(f"Saving data after removing invalid relations...")
        # self.save_relation_data()

    def process_task_data(self):
        try:
            print(f"Processing task data...")
            url = "https://huggingface.co/api/tasks"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                for task in data:
                    if data[task]['id'] not in self.task_ids:
                        # logging.info(f"[task] Found task: {data[task]['id']} - {data[task]['label']}")
                        self.task_ids.add(data[task]['id'])
                        self.processed_tasks.append({
                            'id': data[task]['id'],
                            'label': data[task]['label']
                        })
            else:
                logging.error(f"[task] Failed to fetch task details, status code: {response.status_code}")
            data = get_model_tags()
            if 'pipeline_tag' in data:
                for tag in data['pipeline_tag']:
                    if tag['id'] not in self.task_ids:
                        # logging.info(f"[task] Found task: {tag['id']} - {tag['label']}")
                        self.task_ids.add(tag['id'])
                        self.processed_tasks.append({
                            'id': tag['id'],
                            'label': tag['label']
                        })
            data = get_dataset_tags()
            if 'task_categories' in data:
                for tag in data['task_categories']:
                    tag_id = tag['id'].split(':')[1]
                    if tag_id not in self.task_ids:
                        # logging.info(f"[task] Found task: {tag_id} - {tag['label']}")
                        self.task_ids.add(tag_id)
                        self.processed_tasks.append({
                            'id': tag_id,
                            'label': tag['label']
                        })
            
            print(f"Saving task data...")
            self.save_data('tasks', self.processed_tasks)
        except Exception as e:
            logging.error(f"[task] Exception occurred while fetching task details, error: {e}")
            return None

    def get_tag_classification(self, tag_type):
        if tag_type == 'model':
            data = get_model_tags()
        elif tag_type == 'dataset':
            data = get_dataset_tags()
        else:
            raise ValueError(f"[{tag_type}] Invalid tag type for classification")

        tag_mapping = {}
        for tag_type in data:
            for tag in data[tag_type]:
                tag_mapping[tag['id']] = {
                    'type': tag['type'],
                    'label': tag['label']
                }
        return tag_mapping

    def get_model_details(self, model, tag_mapping):
        # logging.info(f"[model] Processing details for model: {model.id}")
        model_data = {
            'id': model.id,
            'name': model.id.split('/')[1],
            'createdAt': model.created_at,
            'lastModified': model.last_modified,
            'downloads': model.downloads,
            'likes': model.likes,
            'region': None,
            'other': [],
            'libraries': [],
            'license': None,
            'languages': [],
            'pipeline_tag': [],
            'description': None
        }

        self.username_ids.add(model.author)
        self.username_publish_model.append({
            'username': model.author,
            'model_id': model.id
        })
        # logging.info(f"[model] {model.id} is published by: {model.author}")

        if model.pipeline_tag is not None:
            if model.pipeline_tag in tag_mapping:
                model_data['pipeline_tag'].append(tag_mapping[model.pipeline_tag]['label'])
            else:
                model_data['pipeline_tag'].append(model.pipeline_tag)
                logging.warning(f"[model] {model.id} Pipeline tag has no classification: {model.pipeline_tag}")
            self.model_definedFor_task.append({
                'model_id': model.id,
                'task_id': model.pipeline_tag
            })
        else:
            logging.warning(f"[model] Pipeline tag not found for model: {model.id}")
        if model.tags is not None:
            for tag in model.tags:
                if tag.startswith("dataset:"):
                    dataset_id = tag.split(':')[1]
                    self.model_trainedOrFineTunedOn_dataset.append({
                        'model_id': model.id,
                        'dataset_id': dataset_id
                    })
                    # logging.info(f"[model] {model.id} is trained/Fine-tuned on dataset: {dataset_id}")
                elif tag.startswith("arxiv:"):
                    arxiv_id = tag.split(':')[1]
                    self.arxiv_ids.add(arxiv_id)
                    self.model_cite_paper.append({
                        'model_id': model.id,
                        'arxiv_id': arxiv_id
                    })
                    # logging.info(f"[model] {model.id} cite Arxiv: {arxiv_id}")
                elif tag.startswith("base_model:"):
                    parts = tag.split(":")
                    if len(parts) == 3:
                        base_model_id = parts[2]
                        relation = parts[1]
                        if relation == 'adapter':
                            self.model_adapter_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is adapter model based on: {base_model_id}")
                        elif relation == 'finetune':
                            self.model_finetune_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is fine-tuned model based on: {base_model_id}")
                        elif relation == 'merge':
                            self.model_merge_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is merged model based on: {base_model_id}")
                        elif relation == 'quantized':
                            self.model_quantized_model.append({
                                'base_model_id': base_model_id,
                                'model_id': model.id
                            })
                            # logging.info(f"[model] {model.id} is quantized model based on: {base_model_id}")
                        else:
                            # logging.error(f"[model] {model.id} has unknown relation: {relation} for base model: {base_model_id}")
                            pass
                else:
                    if tag in tag_mapping:
                        tag_type = tag_mapping[tag]['type']
                        tag_label = tag_mapping[tag]['label']
                    else:
                        tag_type = 'other'
                        tag_label = tag
                    if tag_type == 'region':
                        model_data['region'] = tag_label
                    elif tag_type == 'other':
                        model_data['other'].append(tag_label)
                    elif tag_type == 'library':
                        model_data['libraries'].append(tag_label)
                    elif tag_type == 'license':
                        model_data['license'] = tag_label
                    elif tag_type == 'language':
                        model_data['languages'].append(tag_label)
                    elif tag_type == 'dataset':
                        pass
                    elif tag_type == 'pipeline_tag':
                        if model.pipeline_tag is None or model.pipeline_tag != tag:
                            model_data['pipeline_tag'].append(tag_label)
                            self.model_definedFor_task.append({
                                'model_id': model.id,
                                'task_id': tag
                            })
                    else:
                        logging.error(f"[model] {model.id} has unknown tag type: {tag_type}")
        else:
            logging.warning(f"[model] Tags not found for model: {model.id}")

        return model_data

    def save_model_data(self):
        self.save_data('models', self.processed_models)
        self.save_data('model_definedFor_task', self.model_definedFor_task)
        self.save_data('model_adapter_model', self.model_adapter_model)
        self.save_data('model_finetune_model', self.model_finetune_model)
        self.save_data('model_merge_model', self.model_merge_model)
        self.save_data('model_quantized_model', self.model_quantized_model)
        self.save_data('model_trainedOrFineTunedOn_dataset', self.model_trainedOrFineTunedOn_dataset)
        self.save_data('model_cite_paper', self.model_cite_paper)

    def process_model_data(self):
        print(f"Preparing model data...")
        tag_mapping = self.get_tag_classification('model')
        models = list_models(full=True, limit=1000)
        
        print(f"Processing models...")
        for model in tqdm(models):
            model_data = self.get_model_details(model, tag_mapping)
            self.processed_models.append(model_data)
        
        # Save model related data
        print(f"Saving model data...")
        self.save_model_data()
        self.model_ids = {model['id'] for model in self.processed_models}

    def get_model_description(self, model):
        try:
            # logging.info(f"[model] Fetching description for model: {model['id']}")
            url = f"https://huggingface.co/{model['id']}/resolve/main/README.md"
            head_response = self.session.head(url, allow_redirects=True)
            if head_response.status_code == 200:
                content_length = int(head_response.headers.get('Content-Length', 0))
                if content_length > 100 * 1024 * 1024:  # 100MB
                    logging.warning(f"[model] Description for model {model['id']} is too large ({content_length} bytes), skipping.")
                    return
                response = self.session.get(url)
                if response.status_code == 200:
                    model['description'] = response.text
            elif head_response.status_code == 404:
                logging.warning(f"[model] Description not found for model: {model['id']}")
            else:
                logging.error(f"[model] Failed to fetch description for model: {model['id']}, status code: {head_response.status_code}")
        except Exception as e:
            logging.error(f"[model] Exception occurred while fetching description for model: {model['id']}, error: {e}")

    def process_model_description(self):
        print(f"Fetching model descriptions...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.get_model_description, self.processed_models),
                total=len(self.processed_models)
            ))
        
        # Save model related data
        print(f"Saving model data...")
        self.save_data('models', self.processed_models)

    def get_dataset_details(self, dataset, tag_mapping):
        # logging.info(f"[dataset] Processing details for dataset: {dataset.id}")
        dataset_data = {
            'id': dataset.id,
            'name': dataset.id.split('/')[1],
            'createdAt': dataset.created_at,
            'lastModified': dataset.last_modified,
            'downloads': dataset.downloads,
            'likes': dataset.likes,
            'libraries': [],
            'license': None,
            'languages': [],
            'other': [],
            'sub-tasks': [],
            'tasks': [],
            'size': None,
            'formats': [],
            'modalities': [],
            'tags': [],
            'description': None
        }

        self.username_ids.add(dataset.author)
        self.username_publish_dataset.append({
            'username': dataset.author,
            'dataset_id': dataset.id
        })
        # logging.info(f"[dataset] {dataset.id} is published by: {dataset.author}")

        if dataset.tags is not None:
            for tag in dataset.tags:
                if tag.startswith("arxiv:"):
                    arxiv_id = tag.split(':')[1]
                    self.arxiv_ids.add(arxiv_id)
                    self.dataset_cite_paper.append({
                        'dataset_id': dataset.id,
                        'arxiv_id': arxiv_id
                    })
                    # logging.info(f"[dataset] {dataset.id} cite Arxiv: {arxiv_id}")
                else:
                    if tag.startswith("task_categories:"):
                        task_id = tag.split(':')[1]
                        self.dataset_definedFor_task.append({
                            'dataset_id': dataset.id,
                            'task_id': task_id
                        })
                    if tag in tag_mapping:
                        tag_type = tag_mapping[tag]['type']
                        tag_label = tag_mapping[tag]['label']
                    else:
                        tag_type = 'tag'
                        tag_label = tag
                    if tag_type == 'library':
                        dataset_data['libraries'].append(tag_label)
                    elif tag_type == 'license':
                        dataset_data['license'] = tag_label
                    elif tag_type == 'language':
                        dataset_data['languages'].append(tag_label)
                    elif tag_type == 'other':
                        dataset_data['other'].append(tag_label)
                    elif tag_type == 'task_ids':
                        dataset_data['sub-tasks'].append(tag_label)
                    elif tag_type == 'task_categories':
                        dataset_data['tasks'].append(tag_label)
                    elif tag_type == 'size_categories':
                        dataset_data['size'] = tag_label
                    elif tag_type == 'format':
                        dataset_data['formats'].append(tag_label)
                    elif tag_type == 'modality':
                        dataset_data['modalities'].append(tag_label)
                    elif tag_type == 'tag':
                        dataset_data['tags'].append(tag_label)
                    else:
                        logging.error(f"[dataset] {dataset.id} has unknown tag type: {tag_type}")
        else:
            logging.warning(f"[dataset] Tags not found for dataset: {dataset.id}")

        return dataset_data

    def save_dataset_data(self):
        self.save_data('datasets', self.processed_datasets)
        self.save_data('dataset_definedFor_task', self.dataset_definedFor_task)
        self.save_data('dataset_cite_paper', self.dataset_cite_paper)

    def process_dataset_data(self):
        print(f"Preparing dataset data...")
        tag_mapping = self.get_tag_classification('dataset')
        datasets = list_datasets(full=True, limit=1000)

        print(f"Processing datasets...")
        for dataset in tqdm(datasets):
            dataset_data = self.get_dataset_details(dataset, tag_mapping)
            self.processed_datasets.append(dataset_data)
        
        # Save dataset related data
        print(f"Saving dataset data...")
        self.save_dataset_data()
        self.dataset_ids = {dataset['id'] for dataset in self.processed_datasets}

    def get_dataset_description(self, dataset):
        try:
            # logging.info(f"[dataset] Fetching description for dataset: {dataset['id']}")
            url = f"https://huggingface.co/datasets/{dataset['id']}/resolve/main/README.md"
            head_response = self.session.head(url, allow_redirects=True)
            if head_response.status_code == 200:
                content_length = int(head_response.headers.get('Content-Length', 0))
                if content_length > 100 * 1024 * 1024:  # 100MB
                    logging.warning(f"[dataset] Description for dataset {dataset['id']} is too large ({content_length} bytes), skipping.")
                    return
                response = self.session.get(url)
                if response.status_code == 200:
                    dataset['description'] = response.text
            elif head_response.status_code == 404:
                logging.warning(f"[dataset] Description not found for dataset: {dataset['id']}")
            else:
                logging.error(f"[dataset] Failed to fetch description for dataset: {dataset['id']}, status code: {head_response.status_code}")
        except Exception as e:
            logging.error(f"[dataset] Exception occurred while fetching description for dataset: {dataset['id']}, error: {e}")

    def process_dataset_description(self):
        print(f"Fetching dataset descriptions...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.get_dataset_description, self.processed_datasets),
                total=len(self.processed_datasets)
            ))
        
        # Save dataset related data
        print(f"Saving dataset data...")
        self.save_data('datasets', self.processed_datasets)

    def get_space_details(self, space_id):
        try:
            # logging.info(f"[space] Processing details for space: {space_id}")
            url = f"https://huggingface.co/api/spaces/{space_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                space_data = {
                    'id': data['id'],
                    'name': data['id'].split('/')[1],
                    'createdAt': data['created_at'] if 'created_at' in data else None,
                    'lastModified': data['last_modified'] if 'last_modified' in data else None,
                    'likes': data['likes'] if 'likes' in data else 0,
                    'tags': []
                }

                space_publisher = data['author']
                self.username_ids.add(space_publisher)
                self.username_publish_space.append({
                    'username': space_publisher,
                    'space_id': data['id']
                })
                # logging.info(f"[space] {space_id} is published by: {space_publisher}")

                if 'tags' in data:
                    for tag in data['tags']:
                        space_data['tags'].append(tag)
                else:
                    logging.warning(f"[space] Tags not found for space: {space_id}")
                if 'models' in data:
                    for model_id in data['models']:
                        self.space_use_model.append({
                            'space_id': data['id'],
                            'model_id': model_id
                        })
                        # logging.info(f"[space] {space_id} uses model: {model_id}")
                if 'datasets' in data:
                    for dataset_id in data['datasets']:
                        self.space_use_dataset.append({
                            'space_id': data['id'],
                            'dataset_id': dataset_id
                        })
                        # logging.info(f"[space] {space_id} uses dataset: {dataset_id}")
                return space_data
            else:
                logging.error(f"[space] Failed to fetch space details for {space_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[space] Exception occurred while fetching space details for {space_id}, error: {e}")
            return None

    def save_space_data(self):
        self.save_data('spaces', self.processed_spaces)
        self.save_data('space_use_model', self.space_use_model)
        self.save_data('space_use_dataset', self.space_use_dataset)

    def process_space_data(self):
        print(f"Preparing space data...")
        spaces = list_spaces(limit=1000)
        space_ids = [space.id for space in tqdm(spaces)]
        
        print(f"Processing spaces...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_space_details, space_ids),
                total=len(space_ids)
            ))

        self.processed_spaces = [space for space in results if space is not None]
        
        # Save space related data
        print(f"Saving space data...")
        self.save_space_data()
        self.space_ids = {space['id'] for space in self.processed_spaces}

    def get_collection_details(self, collection_slug):
        try:
            # logging.info(f"[collection] Processing details for collection: {collection_slug}")
            url = f"https://huggingface.co/api/collections/{collection_slug}"
            response = self.session.get(url, headers={'Authorization': None})
            if response.status_code == 200:
                data = response.json()
                collection_data = {
                    'slug': data['slug'],
                    'title': data['title'],
                    'upvotes': data['upvotes'] if 'upvotes' in data else 0,
                    'description': None
                }

                collection_owner = data['owner']['name']
                collection_owner_type = data['owner']['type']
                self.username_ids.add(collection_owner)

                if collection_owner_type == 'user':
                    self.user_ids.add(collection_owner)
                    self.user_own_collection.append({
                        'user_id': collection_owner,
                        'collection_slug': data['slug']
                    })
                    # logging.info(f"[collection] {collection_slug} is owned by user: {collection_owner}")
                elif collection_owner_type == 'org':
                    self.org_ids.add(collection_owner)
                    self.org_own_collection.append({
                        'org_id': collection_owner,
                        'collection_slug': data['slug']
                    })
                    # logging.info(f"[collection] {collection_slug} is owned by org: {collection_owner}")
                else:
                    logging.error(f"[collection] Unknown owner type: {collection_owner_type} for collection: {collection_slug}")

                if 'description' in data:
                    collection_data['description'] = data['description']
                else:
                    logging.warning(f"[collection] Description not found for collection: {collection_slug}")
                
                if 'items' in data:
                    for item in data['items']:
                        if item['type'] == 'model':
                            self.collection_contain_model.append({
                                'collection_slug': data['slug'],
                                'model_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains model: {item['id']}")
                        elif item['type'] == 'dataset':
                            self.collection_contain_dataset.append({
                                'collection_slug': data['slug'],
                                'dataset_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains dataset: {item['id']}")
                        elif item['type'] == 'space':
                            self.collection_contain_space.append({
                                'collection_slug': data['slug'],
                                'space_id': item['id']
                            })
                            # logging.info(f"[collection] {collection_slug} contains space: {item['id']}")
                        elif item['type'] == 'paper':
                            self.collection_contain_paper.append({
                                'collection_slug': data['slug'],
                                'paper_id': item['id']
                            })
                            self.arxiv_ids.add(item['id'])
                            # logging.info(f"[collection] {collection_slug} contains paper: {item['id']}")
                        else:
                            logging.error(f"[collection] Unknown item type: {item['type']} for collection: {collection_slug}")
                else:
                    logging.warning(f"[collection] Items not found for collection: {collection_slug}")
                return collection_data
            else:
                logging.error(f"[collection] Failed to fetch collection details for {collection_slug}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[collection] Exception occurred while fetching collection details for {collection_slug}, error: {e}")
            return None

    def save_collection_data(self):
        self.save_data('collections', self.processed_collections)
        self.save_data('collection_contain_model', self.collection_contain_model)
        self.save_data('collection_contain_dataset', self.collection_contain_dataset)
        self.save_data('collection_contain_space', self.collection_contain_space)
        self.save_data('collection_contain_paper', self.collection_contain_paper)
        self.save_data('user_own_collection', self.user_own_collection)
        self.save_data('org_own_collection', self.org_own_collection)

    def get_collection_slugs(self, limit=None):
        try:
            # logging.info(f"[collection] Fetching collection slugs...")
            collection_slugs = []
            url = "https://huggingface.co/api/collections"
            response = self.session.get(url, params={'limit': 100}, headers={'Authorization': None})
            if response.status_code != 200:
                logging.error(f"[collection] Failed to fetch collection slugs from {url}, status code: {response.status_code}")
                return []
            collection_slugs.extend([collection.get('slug') for collection in response.json()])
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None and (limit is None or len(collection_slugs) < limit):
                response = self.session.get(next_page, headers={'Authorization': None})
                if response.status_code != 200:
                    logging.error(f"[collection] Failed to fetch collection slugs from {url}, so far fetched: {len(collection_slugs)}, status code: {response.status_code}")
                    return collection_slugs
                collection_slugs.extend([collection.get('slug') for collection in response.json()])
                next_page = response.links.get('next', {}).get('url')
            return collection_slugs if limit is None else collection_slugs[:limit]
        except Exception as e:
            logging.error(f"[collection] Exception occurred while fetching collection slugs, error: {e}")
            return []

    def process_collection_data(self):
        print(f"Preparing collection data...")
        collection_slugs = self.get_collection_slugs(limit=100)

        print(f"Processing collections...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_collection_details, collection_slugs),
                total=len(collection_slugs)
            ))
        
        self.processed_collections = [collection for collection in results if collection is not None]
        
        # Save collection related data
        print(f"Saving collection data...")
        self.save_collection_data()
        self.collection_slugs = {collection['slug'] for collection in self.processed_collections}

    def get_paper_details(self, arxiv_id):
        try:
            # logging.info(f"[paper] Processing details for paper: {arxiv_id}")
            url = f"https://huggingface.co/api/papers/{arxiv_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                paper_data = {
                    'id': data['id'],
                    'authors': [author['name'] for author in data['authors']] if 'authors' in data else [],
                    'publishedAt': data['publishedAt'] if 'publishedAt' in data else None,
                    'title': data['title'] if 'title' in data else None,
                    'summary': data['summary'] if 'summary' in data else None,
                    'upvotes': data['upvotes'] if 'upvotes' in data else 0
                }
                if 'authors' in data:
                    for author in data['authors']:
                        if 'user' in author:
                            user_id = author['user']['user']
                            self.user_ids.add(user_id)
                            self.user_publish_paper.append({
                                'user_id': user_id,
                                'paper_id': data['id']
                            })
                            # logging.info(f"[paper] {arxiv_id} has a registered author: {user_id}")
                return paper_data
            else:
                logging.error(f"[paper] Failed to fetch paper details for {arxiv_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[paper] Exception occurred while fetching paper details for {arxiv_id}, error: {e}")
            return None

    def save_paper_data(self):
        self.save_data('papers', self.processed_papers)
        self.save_data('user_publish_paper', self.user_publish_paper)

    def process_paper_data(self):
        print(f"Processing papers...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_paper_details, self.arxiv_ids),
                total=len(self.arxiv_ids)
            ))
        
        self.processed_papers = [paper for paper in results if paper is not None]

        # Save paper related data
        print(f"Saving paper data...")
        self.save_paper_data()
        self.paper_ids = {paper['id'] for paper in self.processed_papers}
    
    def get_user_like_by_repo(self, repo_id, repo_type):
        try:
            # logging.info(f"[user_like] Processing like for {repo_type}: {repo_id}")
            user_like_repo = []
            url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/likers"
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"[user_like] Failed to fetch likes for {repo_type}: {repo_id}, status code: {response.status_code}")
                return []
            for user in response.json():
                self.user_ids.add(user['user'])
                user_like_repo.append({
                    'user_id': user['user'],
                    f'{repo_type}_id': repo_id
                })
                # logging.info(f"[user_like] {user['user']} liked {repo_type}: {repo_id}")
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[user_like] Failed to fetch likes for {repo_type}: {repo_id} from {url}, so far fetched: {len(user_like_repo)}, status code: {response.status_code}")
                    return user_like_repo
                for user in response.json():
                    self.user_ids.add(user['user'])
                    user_like_repo.append({
                        'user_id': user['user'],
                        f'{repo_type}_id': repo_id
                    })
                    # logging.info(f"[user_like] {user['user']} liked {repo_type}: {repo_id}")
                next_page = response.links.get('next', {}).get('url')
            return user_like_repo
        except Exception as e:
            logging.error(f"[user_like] Exception occurred while fetching likes for {repo_type}: {repo_id}, error: {e}")
            return []

    def save_user_like_data(self):
        self.save_data('user_like_model', self.user_like_model)
        self.save_data('user_like_dataset', self.user_like_dataset)
        self.save_data('user_like_space', self.user_like_space)

    def process_user_like_repo(self):
        for repo_type, repo_ids, result_list in [
            ('model', self.model_ids, self.user_like_model),
            ('dataset', self.dataset_ids, self.user_like_dataset),
            ('space', self.space_ids, self.user_like_space)
        ]:
            print(f"Processing likes for {repo_type}s...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(lambda repo_id: self.get_user_like_by_repo(repo_id, repo_type), repo_ids),
                    total=len(repo_ids)
                ))
            
            for result in results:
                result_list.extend(result)
        
        # Save user like related data
        print(f"Saving user like data...")
        self.save_user_like_data()

    def username_classification(self, username):
        try:
            # logging.info(f"[user&org] Classifying user/organization: {username}")
            if username in self.user_ids or username in self.org_ids:
                return
            url = f"https://huggingface.co/api/users/{username}/overview"
            response = self.session.get(url)
            if response.status_code not in [200, 404]:
                logging.error(f"[user&org] Failed to classify user/organization: {username} as user, status code: {response.status_code}")
                return
            if response.status_code == 200:
                self.user_ids.add(username)
            url = f"https://huggingface.co/api/organizations/{username}/overview"
            response = self.session.get(url)
            if response.status_code not in [200, 404]:
                logging.error(f"[user&org] Failed to classify user/organization: {username} as organization, status code: {response.status_code}")
                return
            if response.status_code == 200:
                self.org_ids.add(username)
            if username not in self.user_ids and username not in self.org_ids:
                logging.error(f"[user&org] Failed to classify user/organization: {username} as user/organization")
        except Exception as e:
            logging.error(f"[user&org] Exception occurred while fetching user details for {username}, error: {e}")
        
    def get_user_overview(self, user_id):
        try:
            # logging.info(f"[user] Processing details for user: {user_id}")
            url = f"https://huggingface.co/api/users/{user_id}/overview"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                user_data = {
                    'id': data['user'],
                    'fullname': data['fullname'] if 'fullname' in data else None
                }
                return user_data
            else:
                logging.error(f"[user] Failed to fetch user details for {user_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[user] Exception occurred while fetching user details for {user_id}, error: {e}")
            return None

    def get_org_overview(self, org_id):
        try:
            # logging.info(f"[org] Processing details for org: {org_id}")
            url = f"https://huggingface.co/api/organizations/{org_id}/overview"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                org_data = {
                    'id': data['name'],
                    'fullname': data['fullname'] if 'fullname' in data else None
                }
                return org_data
            else:
                logging.error(f"[org] Failed to fetch org details for {org_id}, status code: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"[org] Exception occurred while fetching org details for {org_id}, error: {e}")
            return None
        
    def get_user_followers(self, user_id):
        try:
            # logging.info(f"[user] Processing followers for user: {user_id}")
            user_followers = []
            url = f"https://huggingface.co/api/users/{user_id}/followers"
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"[user] Failed to fetch followers for user: {user_id}, status code: {response.status_code}")
                return []
            for follower in response.json():
                # self.user_ids.add(follower['user'])
                user_followers.append({
                    'follower_id': follower['user'],
                    'followee_id': user_id
                })
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[user] Failed to fetch followers for user: {user_id} from {url}, so far fetched: {len(user_followers)}, status code: {response.status_code}")
                    return user_followers
                for follower in response.json():
                    # self.user_ids.add(follower['user'])
                    user_followers.append({
                        'follower_id': follower['user'],
                        'followee_id': user_id
                    })
                next_page = response.links.get('next', {}).get('url')
            return user_followers
        except Exception as e:
            logging.error(f"[user] Exception occurred while fetching followers for user: {user_id}, error: {e}")
            return []
    
    def get_org_followers(self, org_id):
        try:
            # logging.info(f"[org] Processing followers for org: {org_id}")
            org_followers = []
            url = f"https://huggingface.co/api/organizations/{org_id}/followers"
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"[org] Failed to fetch followers for org: {org_id}, status code: {response.status_code}")
                return []
            for follower in response.json():
                # self.user_ids.add(follower['user'])
                org_followers.append({
                    'user_id': follower['user'],
                    'org_id': org_id
                })
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[org] Failed to fetch followers for org: {org_id} from {url}, so far fetched: {len(org_followers)}, status code: {response.status_code}")
                    return org_followers
                for follower in response.json():
                    #self.user_ids.add(follower['user'])
                    org_followers.append({
                        'user_id': follower['user'],
                        'org_id': org_id
                    })
                next_page = response.links.get('next', {}).get('url')
            return org_followers
        except Exception as e:
            logging.error(f"[org] Exception occurred while fetching followers for org: {org_id}, error: {e}")
            return []
        
    def get_org_members(self, org_id):
        try:
            # logging.info(f"[org] Processing members for org: {org_id}")
            org_members = []
            url = f"https://huggingface.co/api/organizations/{org_id}/members"
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"[org] Failed to fetch members for org: {org_id}, status code: {response.status_code}")
                return []
            for member in response.json():
                # self.user_ids.add(member['user'])
                org_members.append({
                    'user_id': member['user'],
                    'org_id': org_id
                })
            next_page = response.links.get('next', {}).get('url')
            while next_page is not None:
                response = self.session.get(next_page)
                if response.status_code != 200:
                    logging.error(f"[org] Failed to fetch members for org: {org_id} from {url}, so far fetched: {len(org_members)}, status code: {response.status_code}")
                    return org_members
                for member in response.json():
                    # self.user_ids.add(member['user'])
                    org_members.append({
                        'user_id': member['user'],
                        'org_id': org_id
                    })
                next_page = response.links.get('next', {}).get('url')
            return org_members
        except Exception as e:
            logging.error(f"[org] Exception occurred while fetching members for org: {org_id}, error: {e}")
            return []

    def save_user_and_org_data(self):
        self.save_data('users', self.processed_users)
        self.save_data('orgs', self.processed_orgs)
        self.save_data('user_publish_model', self.user_publish_model)
        self.save_data('user_publish_dataset', self.user_publish_dataset)
        self.save_data('user_publish_space', self.user_publish_space)
        # self.save_data('user_publish_paper', self.user_publish_paper)
        # self.save_data('user_own_collection', self.user_own_collection)
        # self.save_data('user_like_model', self.user_like_model)
        # self.save_data('user_like_dataset', self.user_like_dataset)
        # self.save_data('user_like_space', self.user_like_space)
        self.save_data('user_follow_user', self.user_follow_user)
        self.save_data('user_affiliatedWith_org', self.user_affiliatedWith_org)
        self.save_data('user_follow_org', self.user_follow_org)
        self.save_data('org_publish_model', self.org_publish_model)
        self.save_data('org_publish_dataset', self.org_publish_dataset)
        self.save_data('org_publish_space', self.org_publish_space)
        # self.save_data('org_own_collection', self.org_own_collection)

    def process_user_and_org_data(self):
        print(f"Classifying users and organizations...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(self.username_classification, self.username_ids),
                total=len(self.username_ids)
            ))
        
        print(f"Processing organizations...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_overview, self.org_ids),
                total=len(self.org_ids)
            ))
        self.processed_orgs = [org for org in results if org is not None]
        print(f"Processing followers for organizations...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_followers, self.org_ids),
                total=len(self.org_ids)
            ))
        for result in results:
            self.user_follow_org.extend(result)
        print(f"Processing members for organizations...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_org_members, self.org_ids),
                total=len(self.org_ids)
            ))
        for result in results:
            self.user_affiliatedWith_org.extend(result)

        print(f"Processing users...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_user_overview, self.user_ids),
                total=len(self.user_ids)
            ))
        self.processed_users = [user for user in results if user is not None]
        print(f"Processing followers for users...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.get_user_followers, self.user_ids),
                total=len(self.user_ids)
            ))
        for result in results:
            self.user_follow_user.extend(result)

        print(f"Post processing user and org data...")
        for edge_list, user_list, org_list, edge_key in [
            (self.username_publish_model, self.user_publish_model, self.org_publish_model, 'model_id'),
            (self.username_publish_dataset, self.user_publish_dataset, self.org_publish_dataset, 'dataset_id'),
            (self.username_publish_space, self.user_publish_space, self.org_publish_space, 'space_id')
        ]:
            for edge in edge_list:
                username = edge['username']
                if username in self.org_ids:
                    org_list.append({
                        'org_id': edge['username'],
                        edge_key: edge[edge_key]
                    })
                elif username in self.user_ids:
                    user_list.append({
                        'user_id': edge['username'],
                        edge_key: edge[edge_key]
                    })

        # Save user and org related data
        print(f"Saving user and org data...")
        self.save_user_and_org_data()

    def run(self):
        """Main execution method"""
        try:
            self.process_task_data()
            self.process_model_data()
            self.process_model_description()
            self.process_dataset_data()
            self.process_dataset_description()
            self.process_space_data()
            self.process_user_like_repo()
            self.process_collection_data()
            self.process_paper_data()
            self.process_user_and_org_data()
            self.verify_relations()
            self.save_all_data()
            self.log_stats()
            logout()
            logging.info("[constructor] Data fetching completed successfully")
        except Exception as e:
            logging.error(f"[constructor] Error during execution: {str(e)}")
            raise

def main():
    constructor = KGConstructor()
    constructor.run()

if __name__ == "__main__":
    main()