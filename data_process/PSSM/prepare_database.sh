wget -P ./data/customdb wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz

gunzip ./data/customdb/uniprot_sprot.fasta.gz


makeblastdb -in ./data/customdb/customdb.fasta -dbtype prot -out ./data/customdb/customdb
#psiblast -query dataProcess/test.fasta -db ./data/swissprot -evalue 0.001 -num_threads 1 -num_iterations 3 -out_ascii_pssm ./test.pssm