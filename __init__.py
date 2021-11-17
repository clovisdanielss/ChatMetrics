import anvil
import anvil.media
import os
import shutil
from dotenv import load_dotenv
from zipfile import ZipFile
from Dialogflow.get_dialogflow_intents import get_dialogflow_intents
from Dialogflow.dialogflow_converter import dialogflow_converter

load_dotenv(".env")
temp_folder = "./temp/"

@anvil.server.callable
def read_zip(file):
    try:
        delete_temp_folder()
        filename = file.name
        anvil.media.write_to_file(file, filename)
        with ZipFile(filename) as zipfile:
            zipfile.extractall(temp_folder)
        os.remove(filename)
        intents, paths = get_dialogflow_intents(path=temp_folder)
        if len(intents) == 0:
            delete_temp_folder()
            return False
        return intents
    except:
        if file.name in os.listdir("."):
            os.remove(file.name)
        return False

@anvil.server.callable
def read_intent(intent):
    filename = temp_folder + "intents/" + intent
    return dialogflow_converter(filename)


def delete_temp_folder():
    folders = os.listdir(".")
    return None if "temp" not in folders else shutil.rmtree(temp_folder)


@anvil.server.callable
def get_hello():
    return "hello"


def run_server():
    key = os.environ.get("TOKEN")
    anvil.server.connect(key)
    anvil.server.wait_forever()


if __name__ == "__main__":
    run_server()
