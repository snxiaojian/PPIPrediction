import os
import sys
from Bio import SeqIO
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder


if __name__ == '__main__':
    folder = './data/customdb/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    files = get_fasta_names_from_folder(folder)

    ids = set()
    records = []
    for file in files:
        file_path = folder + file
        with open(file_path, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                    records.append(record)
                    ids.add(record.id)

    unique_records = []
    for record in records:
        if record.id in ids:
            unique_records.append(record)
            ids.remove(record.id)
    
    print(len(unique_records))
    print(len(records))
    out_file = './data/customdb/customdb.fasta'
    with open(out_file, "w") as output_handle:
        SeqIO.write(unique_records, output_handle, "fasta")
    cmd = "makeblastdb -in ./data/customdb/customdb.fasta -dbtype prot -out ./data/customdb/customdb"
    print("executing " + cmd)
    statu = os.system(cmd)
