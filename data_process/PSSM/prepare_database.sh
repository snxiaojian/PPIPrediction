wget -P ./data/swissprot ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip ./data/swissprot/uniprot_sprot.fasta.gz
makeblastdb -in ./data/uniprot/uniprot_sprot.fasta -dbtype prot -out ./data/uniprot
psiblast -query dataProcess/test.fasta -db ./data/swissprot -evalue 0.001 -num_threads 1 -num_iterations 3 -out_ascii_pssm ./test.pssm