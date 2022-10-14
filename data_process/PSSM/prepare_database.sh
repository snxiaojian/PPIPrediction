wget -P ./data/swissprot https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz
tar -zxvf swissprot.tar.gz
makeblastdb -in swissprot -dbtype prot -out ./data/swissprot -hash-index
psiblast -query dataProcess/test.fasta -db ./data/swissprot -evalue 0.001 -num_threads 1 -num_iterations 3 -out_ascii_pssm ./test.pssm