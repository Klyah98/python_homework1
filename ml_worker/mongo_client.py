import pickle
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


class MongoDBClient:

    def __init__(self, host, port):
        self.client = MongoClient(host, port)
        if 'api_database' in self.client.list_database_names():
            self.client.drop_database('api_database')
        db = self.client['api_database']
        self.collection = db['models']

    def create(self, model_name: str, model):
        obj = {
            'model_name': model_name,
            'model': pickle.dumps(model),
        }
        try:
            if not self.collection.find_one({'model_name': model_name}):
                self.collection.insert_one(obj)
            return True
        except DuplicateKeyError:
            return False

    def read(self, model_name: str):
        result = self.collection.find_one({'model_name': model_name})
        return result

    def update(self, model_name: str, model):
        filter_old_obj = {'model_name': model_name}
        new_obj = {'model': pickle.dumps(model)}
        self.collection.update_one(filter_old_obj, {'$set': new_obj})

    def delete(self, model_name: str):
        self.collection.delete_one({'model_name': model_name})

    def get_documents_info(self):
        models = ''
        all_doc = self.collection.find({})
        for document in all_doc:
            models += (document['model_name'] + '\n')
        return models
