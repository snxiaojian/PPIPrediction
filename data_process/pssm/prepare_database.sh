if [ ! -f "./data/customdb/uniprot_sprot.fasta" ]; then
    wget -P ./data/customdb wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
    gunzip ./data/customdb/uniprot_sprot.fasta.gz
fi
# query in https://www.ncbi.nlm.nih.gov/taxonomy
declare -A taxid_dict
taxid_dict=([9606]="human" [7227]="fly" [559292]="yeast" [3702]="arabidopsis" [6239]="C-elegans" [316407]="Ecoli-K12-W3110" [284812]="fission-yeast" [2697049]="SARS-CoV-2" [39947]="rice" [4577]="maize" [3818]="peanut", [10090]="mouse")

# Download fasta files from uniprot
for id in "${!taxid_dict[@]}"
do
    filename="./data/customdb/${taxid_dict[$id]}.fasta"
    if [ -f $filename ]; then
        echo "Removing $filename"
        rm -rf $filename
    fi
    echo "Downloading ${taxid_dict[$id]} from uniprot"
    wget "https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=taxonomy_id%3A$id" -O $filename
done

# makeblastdb -in ./data/customdb/customdb.fasta -dbtype prot -out ./data/customdb/customdb
#psiblast -query dataProcess/test.fasta -db ./data/swissprot -evalue 0.001 -num_threads 1 -num_iterations 3 -out_ascii_pssm ./test.pssm