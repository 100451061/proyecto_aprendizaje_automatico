#!/bin/bash

read -p "Introduce un mensaje de commit: " mensaje


git add .
git commit -m "$mensaje"
# Se suben los cambios al repositorio remoto con el parámetro push
git push origin