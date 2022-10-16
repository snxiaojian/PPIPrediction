wget -P ./data/go http://purl.obolibrary.org/obo/go.obo
wget -P ./data/go http://geneontology.org/gene-associations/goa_human.gaf.gz
wget -P ./data/go http://current.geneontology.org/annotations/tair.gaf.gz
wget -P ./data/go http://current.geneontology.org/annotations/sgd.gaf.gz
wget -P ./data/go http://current.geneontology.org/annotations/fb.gaf.gz
cd ./data/go
gunzip *.gaf.gz

