from collections import defaultdict
import json


class GOAParser:
    def __init__(self, goa_files):
        self.anotation = {}
        for goa_file in GOAParser.goa_files:
            self.anotation.update(self.parse(goa_file))
 
    @staticmethod
    def parsed_annotation():
        goa_files: list = [
            './data/go/goa_human.gaf',
            './data/go/fb.gaf',
            './data/go/sgd.gaf',
            './data/go/tair.gaf',
        ]
        return GOAParser(goa_files).anotation

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
