from collections import defaultdict
import json


class GOAParser:
    def __init__(self, goa_files):
        self.annotations = {}
        for goa_file in goa_files:
            self.annotations.update(self.parse(goa_file))

    @staticmethod
    def parsed_annotation():
        goa_files: list = [
            './data/go/goa_human.gaf',
            './data/go/fb.gaf',
            './data/go/sgd.gaf',
            './data/go/tair.gaf',
        ]
        return GOAParser(goa_files).annotations

    def parse(self, file: str):
        annotations = defaultdict(list)
        with open(file, 'rt') as fp:
            for line in fp:
                if line.startswith('!'):
                    continue
                entries = line.strip().split('\t')
                qualifier = entries[3]
                if "NOT" not in qualifier:
                    key = entries[2]
                    go_term = entries[4]
                    if go_term not in annotations[key]:
                        annotations[key].append(go_term)
        return annotations