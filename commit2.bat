#!/bin/bash

# Chemin vers votre dépôt local
REPO_PATH="C:/Users/florian1911/Documents/Streamlit" # Mettez votre chemin ici

# URL de votre dépôt GitHub
REPO_URL="https://github.com/Florian1911/Streamlit.git"

# Message de commit
COMMIT_MESSAGE="Mise à jour automatique"

echo "Chemin du dépôt local: $REPO_PATH"
echo "URL du dépôt GitHub: $REPO_URL"

# Aller dans le répertoire du dépôt
cd "$REPO_PATH" || { echo "Erreur: Impossible d'accéder au répertoire $REPO_PATH"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
echo "Répertoire du dépôt atteint."

# Vérifier si la branche "main" existe et y passer
if git rev-parse --verify --quiet origin/main; then
  git checkout main || { echo "Erreur: Impossible de passer à la branche main"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
  echo "Branche main atteinte."
else
  git checkout -b main origin/master || { echo "Erreur: Impossible de créer et passer à la branche main"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
  echo "Branche main créée et atteinte."
fi

# Ajouter tous les fichiers modifiés et non suivis
git add . || { echo "Erreur: Impossible d'ajouter les fichiers"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
echo "Fichiers ajoutés."

# Vérifier s'il y a des modifications à commiter
if [[ -n $(git status --porcelain) ]]; then
  # Commiter les modifications
  git commit -m "$COMMIT_MESSAGE" || { echo "Erreur: Impossible de commiter les modifications"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
  echo "Modifications commises."
else
  echo "Aucune modification à commiter."
fi

# Pousser les modifications vers GitHub
git push origin main || { echo "Erreur: Impossible de pousser les modifications"; read -p "Appuyez sur Entrée pour continuer..."; exit 1; }
echo "Modifications poussées."

echo "Mise à jour réussie !"
read -p "Appuyez sur Entrée pour fermer la fenêtre..."