# This is a sample Python script.
import json

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
NAME = "name"
DATA = "data"
TEXT = "text"
META = "meta"
ID = "id"


def dialogflow_converter(filename):
    json_data = None
    json_metadata = None
    with open(filename + ".json", 'r', encoding='utf-8') as file:
        json_metadata = json.load(file)
    with open(filename + "_usersays_pt-br.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    intent, entities = {
               "name": json_metadata[NAME],
               "phrases": extract_training_phrase(json_data),
           }, extract_entity_phrases(json_data)
    with open(filename + "_intent_result.json", "w", encoding='utf-8') as file:
        json.dump(intent, file, ensure_ascii=False)
    with open(filename + "_entity_result.json", "w", encoding='utf-8') as file:
        json.dump(entities, file, ensure_ascii=False)
    return intent, entities


def extract_entity_phrases(json_data):
    phrases = []
    for response in json_data:
        doc = {
            "id": response[ID],
            "entities": None,
            "phrase": "",
            "word": "",
        }
        for part in response[DATA]:
            if META in part.keys():
                doc["entities"] = part[META]
                doc["word"] = part[TEXT]
            doc["phrase"] += part[TEXT]
        if doc["entities"] is not None:
            phrases.append(doc)
    return phrases


def extract_training_phrase(json_data):
    phrases = []
    for response in json_data:
        phrase = ""
        for part in response[DATA]:
            phrase += part[TEXT]
        phrases.append(phrase)
    return phrases