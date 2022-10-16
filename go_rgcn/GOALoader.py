import json

class GOALoader(dict):
    def __init__(self, goa_file_path):
        super(GOALoader, self).__init__()
        self.load(goa_file_path)
    
    GOAPath = './data/go/goa.json'