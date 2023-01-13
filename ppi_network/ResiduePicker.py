
import sys
sys.path.append("./")
from data_process.pssm.PSSMGenerator import PSSMGenerator
from data_process.util import records_from_filtered_input, gene_from_record

class Residue:
    def __init__(self, location, acid, pssm_row):
        self.location = location
        self.acid = acid
        self.pssm_row = pssm_row
        self.acid_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.best_value = max(self.pssm_row)
        self.current_value = self.pssm_row[self.acid_list.index(self.acid)]
    
    
    def is_the_best_acid(self):
        return self.acid == self.best_acid()
        
    def best_acid(self):
        best_acid_index = self.pssm_row.index(self.best_value)
        return self.acid_list[best_acid_index]
    
class ResiduePicker:
    def __init__(self,record, pssm):
        super(ResiduePicker,self).__init__()
        self.sequence = record.seq._data
        self.pssm = pssm
        
    def pick_residue(self, number: int):
        zeroes = [-200000] * number
        residues = []
        length = min(len(self.sequence), self.pssm.shape[0])
        if length < number:
            for i in range(len(self.sequence)):
                residues.append(Residue(i, self.sequence[i], self.pssm[i]))
            residues.sort(key=lambda x: x.current_value)
            result = [r.location for r in residues] + zeroes[len(residues):number]
            return result
        else:
            for i in range(length):
                residues.append(Residue(i, self.sequence[i], self.pssm[i]))

            residues.sort(key=lambda x: x.current_value)
            return [r.location for r in residues][:number]
        
            
    
if __name__ == "__main__":
     records, species_dict = records_from_filtered_input()
     pid = "A8TX70"
     species = species_dict[pid]
     record = records[pid]
     pssm_file = "./data/pssm/" + species + "/" + pid + ".pssm"
     pssm_data = PSSMGenerator.readFromPSSM(pssm_file)
     picker = ResiduePicker(record, pssm_data)
     indexes = picker.pick_residue(number=50)
     print(indexes)