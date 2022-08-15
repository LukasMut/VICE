#!/bin/bash

URLS=(
	"https://raw.githubusercontent.com/LukasMut/VICE/main/data/files/item_names.tsv"
	"https://raw.githubusercontent.com/LukasMut/VICE/main/data/files/things_concepts.tsv"
	"https://raw.githubusercontent.com/LukasMut/VICE/main/embeddings/things/final_embedding.npy"
);

FILES=(
	"item_names.tsv"
	"things_concepts.tsv"
	"final_embedding.npy"
);

dataset="things";
data_dir="$(pwd)/data/${dataset}";
embedding_dir="$(pwd)/embeddings/${dataset}";
subdirs=( $data_dir $embedding_dir );

for subdir in ${subdirs[@]}; do
	if [[ -d $subdir ]]; then
		printf "\n$subdir exists\n"
	else
		mkdir -p "$subdir";
		printf "\nCreated $subdir\n"	
	fi
done


for i in ${!URLS[@]}; do

	file=${FILES[i]};
	url=${URLS[i]};

	if [[ "$i" == "2" ]]; then
		subdir=$embedding_dir;
	else
		subdir=$data_dir;
	fi	
	curl -O "$url";	
	if [[ -f $file ]]; then
		echo "$url successfully downloaded"
		mv "$file" "$subdir";
	else
		echo "$url not successfully downloaded"
		exit -1
	fi
done
cd ..
