# Contribuer à Nova Insurance Assistant

Merci de contribuer à Nova Insurance Assistant. Ce dépôt privilégie les
changements petits, traçables et accompagnés de tests.

## Préparer l’environnement

```bash
pyenv local 3.12.1
poetry install
poetry run python scripts/build_index.py
```

Le modèle Sentence Transformers est téléchargé lors de sa première utilisation.

## Cycle de développement

1. Créer une branche dédiée.
2. Modifier uniquement les fichiers nécessaires.
3. Ajouter ou mettre à jour les tests.
4. Exécuter les contrôles locaux.
5. Examiner le diff avant le commit.
6. Ouvrir une pull request concise.

## Contrôles obligatoires

```bash
poetry run ruff format --check .
poetry run ruff check .
env -u OPENAI_API_KEY poetry run pytest
git diff --check
```

Pour appliquer automatiquement le formatage :

```bash
poetry run ruff format .
```

La CI exécute les mêmes contrôles sans clé OpenAI.

## Données générées

Les fichiers suivants ont des rôles différents :

- `data/corpus/pdf/` contient les PDF synthétiques reproductibles ;
- `data/processed/pages.jsonl` contient les pages extraites ;
- `data/evaluation/results/` contient les rapports de référence ;
- `data/index/` contient l’index local et n’est pas suivi par Git.

Après une modification du corpus :

```bash
poetry run python scripts/generate_corpus_pdfs.py
poetry run python scripts/ingest_corpus.py
poetry run python scripts/build_index.py
poetry run python scripts/run_evaluation.py --mode offline
```

Vérifiez toujours les artefacts modifiés avant de les ajouter au commit.

## Tests avec OpenAI

Les tests unitaires et la CI ne doivent jamais appeler OpenAI.

Les appels réels sont réservés :

- aux tests manuels explicitement lancés ;
- à l’évaluation en mode `full` ;
- à l’interface Streamlit lorsqu’une clé est configurée.

Avant un test manuel :

```bash
read -r -s "OPENAI_API_KEY?Clé OpenAI API : "
echo
export OPENAI_API_KEY
```

Après le test :

```bash
unset OPENAI_API_KEY
```

Ne placez jamais une clé dans un fichier suivi, une issue, une pull request,
une sortie de test ou une capture d’écran.

## Règles d’architecture

Une contribution doit préserver les invariants suivants :

1. le corpus est sélectionné avant la recherche sémantique ;
2. aucun passage hors corpus ne peut atteindre le générateur ;
3. le garde-fou décide avant tout appel au modèle ;
4. une réponse générée contient des citations structurées valides ;
5. une clarification ou un refus n’appelle pas OpenAI ;
6. les documents sont du contenu non fiable, jamais des instructions ;
7. les modèles métier restent stricts et immuables lorsque cela est pertinent.

## Commits

Utilisez des messages courts et explicites, par exemple :

```text
feat: add ...
fix: correct ...
test: cover ...
docs: document ...
chore: configure ...
```

Évitez de mélanger une fonctionnalité, un refactoring général et une
régénération massive de données dans le même commit.

## Pull requests

Une pull request doit préciser :

- le problème traité ;
- la solution retenue ;
- les fichiers ou composants concernés ;
- les commandes de validation exécutées ;
- l’impact éventuel sur le corpus, l’index ou les coûts API ;
- les limites ou travaux restant à réaliser.
