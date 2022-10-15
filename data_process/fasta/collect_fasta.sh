wget https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28%28taxonomy_id%3A9606%29%29 -O ./data/input/human.fasta

wget https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28%28taxonomy_id%3A3702%29%29 -O ./data/input/arabidopsis.fasta

wget https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28Saccharomyces%20cerevisiae%29%20AND%20%28model_organism%3A559292%29 -O ./data/input/yeast.fasta

wget https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28Drosophila%20Melanogaster%29%20AND%20%28model_organism%3A7227%29 -O ./data/input/fly.fasta
