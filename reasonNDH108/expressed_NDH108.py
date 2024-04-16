import pandas as pd
from Bio import SeqIO

def read_expression_ids_from_xlsx(xlsx_file_path):
    # 使用pandas读取xlsx文件的第一列
    df = pd.read_excel(xlsx_file_path, engine='openpyxl', usecols=[0], header=None)
    return df[0].tolist()  # 返回id列表

def read_long_mRNA_ids(txt_file_path):
    # 读取文本文件中的每一行为一个id列表
    with open(txt_file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def select_transcript_id(expression_ids, long_mrna_ids):
    # 选取转录本id
    selected_ids = []
    expression_prefixes = {id.split('.')[0]: id for id in expression_ids}

    for long_id in long_mrna_ids:
        prefix = long_id.split('.')[0]
        # 如果前缀在表达id列表中，选择表达id，否则选择长转录本id
        if prefix in expression_prefixes:
            selected_ids.append(expression_prefixes[prefix])
            print(f"use express {expression_prefixes[prefix]}")
        else:
            selected_ids.append(long_id)
            print(f"use long {long_id}")
    return selected_ids

def write_ids_to_file(ids, file_path):
    # 将id写入新文件
    with open(file_path, 'w') as file:
        for id in ids:
            file.write(id + '\n')

# 文件路径
xlsx_file_path = './reasonNDH108/NDH108.protmap.xlsx'
long_mrna_file_path = './reasonNDH108/NDH108.long.mRNA.id'
output_file_path = './reasonNDH108/NDH108_reconstruct.mRNA.id'  # 输出文件路径

# 执行功能
expression_ids = read_expression_ids_from_xlsx(xlsx_file_path)
long_mrna_ids = read_long_mRNA_ids(long_mrna_file_path)
selected_transcript_ids = select_transcript_id(expression_ids, long_mrna_ids)

# 将选定的ID写入新文件
# write_ids_to_file(selected_transcript_ids, output_file_path)

# print(f"IDs have been written to {output_file_path}")

# # 过滤fasta文件
# filtered_records = (record for record in SeqIO.parse('./data/filtered_input_only_pssm/NDH108.fasta', 'fasta') if record.id in selected_transcript_ids)

# # 写入新的fasta文件
# SeqIO.write(filtered_records, './data/filtered_input_only_pssm/reconstruct_NDH108.fasta', 'fasta')
