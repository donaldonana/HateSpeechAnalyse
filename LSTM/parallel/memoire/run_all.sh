#!/bin/bash
make
echo -e "\nNombre d'execution : "
read exe
rm -rf resultats/*
for ((i=1; i<=$exe; i++))  
do  
echo -e "\n--------------------Exécution $i--------------------\n"  
./app.exe -hiden 80 -epoch 15 -batch 32 -lr 0.1 -thread 1 -execution $i
done 

