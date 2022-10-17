wget -P ./data/biogrid https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.214/BIOGRID-ALL-4.4.214.tab2.zip
wget -P ./data/biogrid https://downloads.thebiogrid.org/Download/BioGRID/External-Database-Builds/UNIPROT.tab.txt
unzip -o -d ./data/biogrid ./data/biogrid/BIOGRID-ALL-4.4.214.tab2.zip
rm -rf ./data/biogrid/BIOGRID-ALL-4.4.214.tab2.zip