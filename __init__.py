import anvil
import anvil.media
import os
import shutil
from dotenv import load_dotenv
from zipfile import ZipFile
from Dialogflow.get_dialogflow_intents import get_dialogflow_intents
from Dialogflow.dialogflow_converter import dialogflow_converter
from Training.training_intent import get_training_model
from Training.training_intent import get_confusion_matrix as confusion_matrix
from uuid import uuid1
from Metrics.metrics import false_negative, false_positives, true_negative, true_positives

load_dotenv(".env")
temp_folder = "./temp/"
models = {}


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


@anvil.server.callable
def get_hello():
    return "hello"


@anvil.server.callable
def get_default_model(intents_json, epochs):
    training = get_training_model(raw_intents_json=intents_json, path=temp_folder, epochs=epochs)
    uuid = str(uuid1())
    models[uuid] = training
    return uuid


@anvil.server.callable
def get_model_history(uuid):
    if uuid in models.keys():
        return models[uuid].history
    else:
        return {}


@anvil.server.callable
def get_confusion_matrix(uuid):
    if uuid in models.keys():
        data = models[uuid].data.name
        classes_names = [data.iloc[i] for i in range(models[uuid].data.shape[0])]
        cmnumpy = confusion_matrix(models[uuid], classes_names).to_numpy()
        cm = []
        for i in range(cmnumpy.shape[0]):
            arr = []
            for j in range(cmnumpy.shape[1]):
                arr.append(int(cmnumpy[i, j]))
            cm.append(arr[::-1])
        return cm
    else:
        return []

@anvil.server.callable
def extract_tfpn(uuid):
    if uuid in models.keys():
        model = models[uuid]
        data = model.data.name
        classes_names = [data.iloc[i] for i in range(model.data.shape[0])]
        cm = confusion_matrix(model, classes_names)
        tp = true_positives(cm, classes_names)
        fp = false_positives(cm, classes_names)
        tn = true_negative(cm, classes_names)
        fn = false_negative(cm, classes_names)
        return tp, fp, tn, fn
    else:
        return [], [], [], []

def run_server():
    key = os.environ.get("TOKEN")
    anvil.server.connect(key)
    anvil.server.wait_forever()


def delete_temp_folder():
    folders = os.listdir(".")
    return None if "temp" not in folders else shutil.rmtree(temp_folder)


if __name__ == "__main__":
    run_server()
