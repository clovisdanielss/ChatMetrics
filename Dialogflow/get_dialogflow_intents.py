import os


def get_dialogflow_intents(path, language="pt-br"):
    files = os.listdir(path)
    has_intent_folder = "intents" in files
    if not has_intent_folder:
        return [],[]
    files = os.listdir(path + "intents/")
    usersays_language = [filename for filename in files if "_usersays_" + language in filename]
    intents = [name.split("_usersays_"+ language)[0] for name in usersays_language]
    usersays_language = [path + "intents/" + filename for filename in usersays_language]
    return intents, usersays_language
