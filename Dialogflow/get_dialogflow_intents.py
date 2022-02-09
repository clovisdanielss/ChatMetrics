import os

from Dialogflow.dialogflow_converter import dialogflow_converter


def get_dialogflow_intents(path, language="pt-br"):
    files = os.listdir(path)
    has_intent_folder = "intents" in files
    if not has_intent_folder:
        return [], []
    files = os.listdir(path + "intents/")
    usersays_language = [filename for filename in files if "_usersays_" + language in filename]
    intents = [name.split("_usersays_" + language)[0] for name in usersays_language]
    usersays_language = [path + "intents/" + filename for filename in usersays_language]
    return intents, usersays_language


def get_dialogflow_intents_as_json(path):
    intents, intents_paths = get_dialogflow_intents(path)
    intents_json = []
    for intent_name in intents:
        intent_json, entity_json = dialogflow_converter(path + "intents/" + intent_name)
        intents_json.append(intent_json)
    return intents_json


def get_dialogflow_entities_as_json(path):
    intents, intents_paths = get_dialogflow_intents(path)
    entities_json = []
    for intent_name in intents:
        intent_json, entity_json = dialogflow_converter(path + "intents/" + intent_name)
        entities_json += entity_json
    return entities_json
