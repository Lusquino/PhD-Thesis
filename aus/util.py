import json

def dictToJSON(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def JSONToDict(filename):
    data = {}
    with open(filename) as f:
        data = json.load(f)
    return data
