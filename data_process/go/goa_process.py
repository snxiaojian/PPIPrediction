from collections import defaultdict
import json


class GOAParser:
    def __init__(self, goa_files: list):
        self.anotation = {}
        for goa_file in goa_files:
            self.anotation.update(self.parse(goa_file))
        self.save_anotation_to_file()

    def save_anotation_to_file(self):
        with open('./data/go/goa.json', 'w') as fp:
            j = json.dumps(self.anotation)
            fp.write(j)

    def parse(self, file: str):
        annotation = defaultdict(list)
        with open(file, 'rt') as fp:
            for line in fp:
                if line.startswith('!'):
                    continue
                entries = line.strip().split('\t')
                qualifier = entries[3]
                if "NOT" not in qualifier:
                    key = entries[2]
                    go_term = entries[4]
                    annotation[key].append(go_term)
        return annotation


if __name__ == "__main__":
    goa_files = [
        './data/go/goa_human.gaf',
        './data/go/fb.gaf',
        './data/go/sgd.gaf',
        './data/go/tair.gaf',
    ]
    goa_parser = GOAParser(goa_files)
