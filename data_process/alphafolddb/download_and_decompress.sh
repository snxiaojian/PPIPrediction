./data_process/alphafolddb/download_3702_from_alphafolddb.sh
./data_process/alphafolddb/download_7227_from_alphafolddb.sh
./data_process/alphafolddb/download_9606_from_alphafolddb.sh
./data_process/alphafolddb/download_559292_from_alphafolddb.sh
cd ./data/alphafolddb/human
for tar in *.tar;  do tar -xvf $tar; done
for json in *.json.gz;  do rm -rf $json; done
for cif in *.cif.gz;  do gunzip $cif; done

cd ../arabidopsis
for tar in *.tar;  do tar -xvf $tar; done
for json in *.json.gz;  do rm -rf $json; done
for cif in *.cif.gz;  do gunzip $cif; done

cd ../fly
for tar in *.tar;  do tar -xvf $tar; done
for json in *.json.gz;  do rm -rf $json; done
for cif in *.cif.gz;  do gunzip $cif; done

cd ../yeast
for tar in *.tar;  do tar -xvf $tar; done
for json in *.json.gz;  do rm -rf $json; done
for cif in *.cif.gz;  do gunzip $cif; done