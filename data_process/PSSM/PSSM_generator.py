import os
import numpy as np
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append("./")
from data_process.util import get_fasta_names_from_folder, id_from_record

class PSSMGenerator:
    temp_fasta_folder = "./data_process/PSSM/temp/"
    db_folder = "./data/customdb/customdb"

    def __init__(self, species):
            records = []
            fasta_file = "./data/input/" + species + ".fasta"
            with open(fasta_file, 'r') as f:
                for record in SeqIO.parse(f, "fasta"):
                    records.append(record)
            self.records = records
            self.species = species
            if not os.path.exists(PSSMGenerator.temp_fasta_folder):
                os.makedirs(PSSMGenerator.temp_fasta_folder)
            self.pssm_folder = "./data/pssm/" + species + "/"
            if not os.path.exists(self.pssm_folder):
                os.makedirs(self.pssm_folder)
    

    def fasta_file_path(self, record):
        return PSSMGenerator.temp_fasta_folder + id_from_record(record)

    def pssm_file_path(self, record):
        return self.pssm_folder + id_from_record(record) + ".pssm"

    def _write_record_to_tmp_file(self,record):
        fasta_path = self.fasta_file_path(record)

        with open(fasta_path, 'w') as f:
            SeqIO.write([record], f, 'fasta')
            print("write to file: " + fasta_path)
        return fasta_path

    def _delete_fasta_temp_file(self,record):
        file = self.fasta_file_path(record)
        os.remove(file)

    @staticmethod
    def get_line_numpy(line ,start=11 , end=91 , step=4):
        '''
        输入一行,返回处理后的np数组
        默认start是11,end是91,step是4
        start=10 end=69 step=3
        '''
        np_line = np.zeros([20])
        index = 0
        for symbol , num in zip(line[start:end:step] , line[start+1:end:step]):
            if symbol == '-' :
                statu = -1
            else:
                statu = 1
            np_line[index] = statu*int(num)
            index = index + 1 
        return np_line   

    @staticmethod
    def readFromPSSM(pssm_file):
        with open(pssm_file , 'r') as f:
            for i in range(2):#前2行无用
                f.readline()
            line = f.readline()#第三行判断
            if line[12] == 'A':
                line = f.readline()
                pssm_np = PSSMGenerator.get_line_numpy( line )
                line = f.readline()
                while line != '\n' :
                    pssm_np = np.vstack( [pssm_np , PSSMGenerator.get_line_numpy( line )] )
                    line = f.readline()
            else:
                line = f.readline()
                pssm_np = PSSMGenerator.get_line_numpy( line ,10,69,3)
                line = f.readline()
                while line != '\n' :
                    pssm_np = np.vstack( [pssm_np , PSSMGenerator.get_line_numpy( line ,10,69,3)] )
                    line = f.readline()
            f.close()
        return pssm_np  

    def _generatePSSMFile(self,record):
        '''
        生成pssm文件
        '''
        fasta_path = self.fasta_file_path(record)
        pssm_file_path = self.pssm_file_path(record)
        print("generating: " + pssm_file_path)
        cmd = 'psiblast -query '+ fasta_path + ' -db ' + PSSMGenerator.db_folder  + ' -evalue 0.001  -num_threads 1 -num_iterations 3 -out_ascii_pssm ' + pssm_file_path  + "> /dev/null 2>&1"
        print("executing " + cmd)
        statu = os.system(cmd)
        assert statu == 0 , '生成PSSM文件失败,cmd命令有错'
    
    def process_with_record(self, record):
        self._write_record_to_tmp_file(record)
        self._generatePSSMFile(record)
        self._delete_fasta_temp_file(record)

    def generate(self):
        recordNeedProcess = []
        for record in self.records:
            pssm_file_path = self.pssm_file_path(record)
            if os.path.exists(pssm_file_path):
                print("pssm file already exists: " + pssm_file_path)
            else:
                recordNeedProcess.append(record)
        with ProcessPoolExecutor() as pool:
            pool.map(self.process_with_record,recordNeedProcess, chunksize=6)


if __name__ == "__main__":
    files = get_fasta_names_from_folder("./data/input")
    for file in files:
        print("processing " + file)
        pssm_generator = PSSMGenerator(file.split(".")[0])
        pssm_generator.generate()