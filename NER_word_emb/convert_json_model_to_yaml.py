from keras.models import model_from_yaml, model_from_json

MODEL_FILE = "./Trash_model_1_128.arch"
WEIGHTS_FILE = "./Trash_model_1_128.weights.h5"

OP_MODEL_FILE = "./Trash_yaml_model.yaml"

print("Reading in model")
with open(MODEL_FILE, 'rb') as ip_file:
    model = model_from_json(ip_file.read())

print("Writing out yaml model")
with open(OP_MODEL_FILE, 'wb') as op_file:
    op_file.write(model.to_yaml())
