from collections import defaultdict
import gzip
import json


class GOAParser:
    def __init__(self, goa_file: str):
        self.goa_file = goa_file
        self.anotation = self.parse(goa_file)
        self.save_anotation_to_file()

    def save_anotation_to_file(self):
        with open('./data/go/goa.json', 'w') as fp:
            j = json.dumps(self.anotation)
            fp.write(j)

    evidence = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC']

    def parse(self, file: str):
        annotation = defaultdict(list)
        index = 0
        with gzip.open(file, 'rt') as fp:
            for line in fp:
                if line.startswith('!'):
                    continue
                entries = line.strip().split('\t')
                qualifier = entries[3]
                evidence_code = entries[6]
                db_object_type = entries[11]
                if db_object_type == "protein" \
                        and "NOT" not in qualifier \
                        and evidence_code in GOAParser.evidence:
                    protein = entries[1]
                    go_term = entries[4]
                    annotation[protein].append(go_term)

                index += 1
                if index % 100000 == 0:
                    print(index)
        return annotation


if __name__ == "__main__":
    goa_file = "./data/go/goa_uniprot_all.gaf.gz"
    goa_parser = GOAParser(goa_file)
