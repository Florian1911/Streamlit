#!/bin/bash

# Chemin vers votre d�p�t local
REPO_PATH="C:/Users/florian1911/Documents/Streamlit" # Mettez votre chemin ici

# URL de votre d�p�t GitHub
REPO_URL="https://github.com/Florian1911/Streamlit.git"

# Message de commit
COMMIT_MESSAGE="Mise � jour automatique"

echo "Chemin du d�p�t local: $REPO_PATH"
echo "URL du d�p�t GitHub: $REPO_URL"

# Aller dans le r�pertoire du d�p�t
cd "$REPO_PATH" || { echo "Erreur: Impossible d'acc�der au r�pertoire $REPO_PATH"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
echo "R�pertoire du d�p�t atteint."

# V�rifier si la branche "main" existe et y passer
if git rev-parse --verify --quiet origin/main; then
  git checkout main || { echo "Erreur: Impossible de passer � la branche main"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
  echo "Branche main atteinte."
else
  git checkout -b main origin/master || { echo "Erreur: Impossible de cr�er et passer � la branche main"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
  echo "Branche main cr��e et atteinte."
fi

# Ajouter tous les fichiers modifi�s et non suivis
git add . || { echo "Erreur: Impossible d'ajouter les fichiers"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
echo "Fichiers ajout�s."

# V�rifier s'il y a des modifications � commiter
if [[ -n $(git status --porcelain) ]]; then
  # Commiter les modifications
  git commit -m "$COMMIT_MESSAGE" || { echo "Erreur: Impossible de commiter les modifications"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
  echo "Modifications commises."
else
  echo "Aucune modification � commiter."
fi

# Pousser les modifications vers GitHub
git push origin main || { echo "Erreur: Impossible de pousser les modifications"; read -p "Appuyez sur Entr�e pour continuer..."; exit 1; }
echo "Modifications pouss�es."

echo "Mise � jour r�ussie !"
read -p "Appuyez sur Entr�e pour fermer la fen�tre..."