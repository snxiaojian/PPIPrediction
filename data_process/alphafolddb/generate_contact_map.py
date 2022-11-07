import Bio.PDB
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue

import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor



def get_center_atom(residue: Residue):
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom


def calc_residue_dist(residue_one: Residue, residue_two: Residue):
    """Returns the C-alpha distance between two residues"""

    c_atom1: str = get_center_atom(residue_one)
    c_atom2: str = get_center_atom(residue_two)
    diff_vector = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    if np.sqrt(np.sum(diff_vector * diff_vector)) < 8:
      return 1
    else:
      return 0


def calc_dist_matrix(chain: Chain):
    newChain = []
    for row, residue in enumerate(chain):
        if is_protein_residue(residue):
            newChain.append(residue)
    residue_len = len(newChain)
    answer = np.zeros((residue_len, residue_len), int)

    for i, residue_one in enumerate(chain):
        for j, residue_two in enumerate(chain):
            if i != j :
                answer[i, j] = calc_residue_dist(residue_one, residue_two)
    for i in range(residue_len):
        answer[i, i] = 0
    return answer

def is_protein_residue(residue: Residue):
    hetfield = residue.get_id()[0]
    hetname = residue.get_resname()
    if hetfield == ' ' and hetname in aa_codes_keys:
        return True
    return False


def calc_contact_map(pdb_path, pdb_id, chain_id):
    pdb_path: str = pdb_path + "AF-" +  pdb_id + '-F1-model_v2.pdb'
    structure: Structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
    chain: Chain = structure[0][chain_id]
    dist_matrix = calc_dist_matrix(chain)
    return dist_matrix


def calc_cif_contact_map(cif_path, cif_id, chain_id):
    cif_path: str = cif_path + "AF-" +  cif_id + '-F1-model_v3.cif'
    structure: Structure = Bio.PDB.MMCIFParser().get_structure(cif_id, cif_path)
    chain: Chain = structure[0][chain_id]
    dist_matrix = calc_dist_matrix(chain)
    return dist_matrix

aa_codes_keys: list[str] = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
}.keys()


def convert_pdb_to_npz(para_list):
    sequence = para_list[0]
    pdb_id = para_list[1]
    pdb_path = para_list[2]
    output_folder = para_list[3]
    index = para_list[4]
    species = para_list[5]

    npz_file = output_folder + pdb_id + '.npz'
    if os.path.exists(npz_file):
        print('skipping ' + species + " :" + index)
    else:
        contact_map = calc_contact_map(pdb_path, pdb_id, 'A')
        
        np.savez(npz_file, contact=contact_map, seq=sequence)
        print(species + ":" + index)


def convert_cif_to_npz(para_list):
    pdb_id = para_list[0]
    pdb_path = para_list[1]
    output_folder = para_list[2]
    index = para_list[3]
    species = para_list[4]

    npz_file = output_folder + pdb_id + '.npz'

    contact_map = calc_cif_contact_map(pdb_path, pdb_id, 'A')
    np.savez(npz_file, contact=contact_map)
    print(species + ":" + index)

def get_subfolder_from_folder(folder):
    return [name for name in os.listdir(folder) if os.path.isdir(folder + name)]

def get_cifnames_from_folder(folder):
    return [name for name in os.listdir(folder) if name.endswith(".cif")]

def parsealphaFoldIDFromFile(filename):
    return filename.split("AF-")[1].split("-")[0]

def main():
    para_list = []
    alphafold_path = "./data/alphafolddb/"
    folders = get_subfolder_from_folder(alphafold_path)
    for name in folders:
        output_folder = 'data/contact_map/' + name + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        species_alphafold_path = alphafold_path + name + '/'
        filenames = get_cifnames_from_folder(species_alphafold_path)
        for j, filename in enumerate(filenames):
            pdb_id = parsealphaFoldIDFromFile(filename)
            para = [pdb_id, species_alphafold_path, output_folder, str(j), name]
            npz_file = output_folder + pdb_id + '.npz'
            if os.path.exists(npz_file):
                print('skipping ' + name + " :" + str(j))
            else:
                para_list.append(para)
    
    with ProcessPoolExecutor() as pool:
        pool.map(convert_cif_to_npz,para_list, chunksize=6)
    
if __name__=="__main__":
    main()