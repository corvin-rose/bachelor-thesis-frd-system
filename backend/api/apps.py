from django.apps import AppConfig

from backend.port.adapter.bert_class import load_bert_model


class FRDAppConfig(AppConfig):
    name = 'api'

    def ready(self):
        print("Loading BERT Model...")
        load_bert_model()
