declare -A taxid_dict
taxid_dict=([9606]="human" [7227]="fly" [559292]="yeast" [3702]="arabidopsis")

# [6239]="C-elegans" [10090]="mouse" [284812]="fission-yeast"
# Download fasta files from uniprot
for id in "${!taxid_dict[@]}"
do
    filename="./data/input/${taxid_dict[$id]}.fasta"
    echo "Downloading ${taxid_dict[$id]} from uniprot"
    wget "https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=taxonomy_id%3A$id" -O $filename
done