# Référentiel métier — Nova Assurances

## 1. Objectif

Ce document constitue la source de vérité métier du projet Nova Insurance
Assistant.

Les contrats synthétiques, les métadonnées, les réponses attendues et le jeu
d’évaluation doivent respecter les règles définies ici.

## 2. Univers métier

- Compagnie : Nova Assurances
- Territoire commercial : Luxembourg
- Clientèle : particuliers
- Langue documentaire : français
- Devise : euro
- Données : entièrement synthétiques

## 3. Produits et versions

| Produit     | Domaine              | Versions     |
| ----------- | -------------------- | ------------ |
| Nova Home   | Assurance habitation | 2024 et 2025 |
| Nova Auto   | Assurance automobile | 2024 et 2025 |
| Nova Travel | Assurance voyage     | 2025         |

## 4. Nova Home

### 4.1 Garanties communes

- incendie affectant le logement ou les biens mobiliers ;
- dégât des eaux accidentel ;
- vol avec effraction constatée ;
- bris de glace ;
- responsabilité civile ;
- relogement temporaire pendant 15 jours au maximum.

### 4.2 Différences entre versions

| Règle                    |               2024 |               2025 |
| ------------------------ | -----------------: | -----------------: |
| Objets de valeur         |            5 000 € |            7 500 € |
| Franchise dégât des eaux |              300 € |              200 € |
| Relogement               |         100 €/jour |         150 €/jour |
| Assistance d’urgence     | 2 interventions/an | 4 interventions/an |
| Validité                 |   01/01–31/12/2024 |   01/01–31/12/2025 |
| Statut                   |            Archivé |              Actif |

### 4.3 Exclusions

- acte intentionnel ;
- défaut d’entretien connu ;
- usure normale ;
- vol sans effraction ;
- travaux structurels non déclarés ;
- biens professionnels dépassant 2 000 € ;
- sinistre survenu hors du Luxembourg.

### 4.4 Obligations

- déclarer un vol sous 48 heures ;
- déclarer les autres sinistres sous 5 jours ouvrés ;
- limiter raisonnablement les dommages ;
- fournir les justificatifs disponibles ;
- déclarer toute modification importante du risque.

### 4.5 Informations absentes

- soins vétérinaires ;
- assurance santé ;
- véhicules ;
- incidents de voyage ;
- valeur de revente d’un logement.

## 5. Nova Auto

### 5.1 Garanties communes

- responsabilité civile ;
- collision ;
- vol ou tentative avec traces constatables ;
- incendie accidentel ;
- bris de glace ;
- assistance et remorquage ;
- véhicule de remplacement après un sinistre couvert.

### 5.2 Différences entre versions

| Règle                    |                      2024 |             2025 |
| ------------------------ | ------------------------: | ---------------: |
| Franchise collision      |                     500 € |            350 € |
| Déclenchement assistance | Plus de 50 km du domicile |  Dès le domicile |
| Véhicule de remplacement |                   5 jours |         10 jours |
| Effets personnels        |                     500 € |          1 000 € |
| Validité                 |          01/01–31/12/2024 | 01/01–31/12/2025 |
| Statut                   |                   Archivé |            Actif |

### 5.3 Étendue géographique

- Luxembourg ;
- Union européenne ;
- Suisse.

Une extension écrite est nécessaire pour les autres territoires.

### 5.4 Exclusions

- conduite sans permis valide ;
- conduite sous alcool ou stupéfiants ;
- course ou compétition ;
- acte intentionnel ;
- usure ou défaut d’entretien ;
- usage commercial non déclaré ;
- véhicule non verrouillé en cas de vol d’effets personnels ;
- sinistre hors de la zone couverte.

### 5.5 Obligations

- déclarer un accident sous 5 jours ouvrés ;
- déclarer un vol sous 48 heures ;
- prévenir la police en cas de vol ou de dommages corporels ;
- transmettre le constat et les justificatifs ;
- attendre l’expertise avant réparation, hors urgence de sécurité ;
- déclarer tout changement d’usage.

### 5.6 Informations absentes

- valeur de revente du véhicule ;
- coût de l’entretien ;
- sanctions du Code de la route ;
- financement ou leasing ;
- véhicule professionnel non déclaré ;
- disponibilité réelle d’un garage.

## 6. Nova Travel

### 6.1 Version

- Version : 2025
- Validité : 01/01/2025–31/12/2025
- Statut : actif

### 6.2 Garanties

| Garantie                 |                   Limite |
| ------------------------ | -----------------------: |
| Frais médicaux d’urgence |                100 000 € |
| Rapatriement médical     | Frais réels après accord |
| Annulation               |       5 000 € par assuré |
| Interruption             |       3 000 € par assuré |
| Perte ou vol de bagages  |                  1 500 € |
| Retard de bagages        |    300 € après 12 heures |
| Retard de transport      |     250 € après 6 heures |
| Responsabilité civile    |                500 000 € |
| Assistance juridique     |                  5 000 € |

### 6.3 Conditions d’application

- voyage privé ;
- monde entier hors pays officiellement exclus ;
- durée maximale de 90 jours consécutifs ;
- souscription avant le départ ;
- événement imprévisible pendant la validité du contrat.

### 6.4 Exclusions

- maladie connue et non stabilisée ;
- voyage malgré un avis médical contraire ;
- sport extrême sans extension ;
- compétition professionnelle ;
- acte intentionnel ;
- alcool ou stupéfiants ;
- guerre ou zone officiellement déconseillée ;
- voyage de plus de 90 jours ;
- bagage laissé sans surveillance ;
- oubli ou perte sans circonstance identifiable ;
- changement d’avis comme seul motif d’annulation.

### 6.5 Obligations

- contacter l’assistance avant les frais importants ;
- déclarer le sinistre sous 5 jours ouvrés ;
- signaler immédiatement un vol aux autorités locales ;
- conserver les factures, billets et justificatifs ;
- obtenir une attestation du transporteur ;
- fournir un certificat médical si nécessaire.

### 6.6 Informations absentes

- visas ;
- vaccins ;
- météo ;
- prix des billets ;
- recommandations touristiques ;
- taux de change ;
- garanties des cartes bancaires.

## 7. Sélection exacte du corpus

### 7.1 Champs de contexte

- `product` : obligatoire ;
- `version` ou `contract_date` : au moins l’un des deux obligatoire ;
- `language` : français par défaut ;
- `document_type` : facultatif.

### 7.2 Sélection par version

Lorsque la version est fournie, le filtre utilise exactement :

- le produit ;
- la version ;
- la langue ;
- éventuellement le type documentaire.

### 7.3 Sélection par date

Un document est applicable lorsque :

```text
effective_from <= contract_date <= effective_to
```

### 7.4 Cohérence entre date et version

Si la version et la date sont fournies, elles doivent désigner le même corpus.

Une incohérence produit le statut `conflicting_context`.

### 7.5 Documents archivés

Un document archivé reste utilisable lorsqu’il correspond à la version
explicitement demandée.

Le statut `archived` ne constitue donc pas, à lui seul, un motif d’exclusion.

### 7.6 Interdiction d’élargissement silencieux

Si aucun document ne correspond aux filtres exacts, le système ne doit jamais :

- retirer le filtre de version ;
- choisir automatiquement une autre version ;
- rechercher dans un autre produit ;
- utiliser automatiquement la version la plus récente ;
- effectuer une recherche globale.

Le résultat attendu est `no_matching_corpus`.

## 8. Ordre obligatoire du traitement

1. Valider le contexte.
2. Appliquer les filtres exacts.
3. Obtenir les identifiants des documents autorisés.
4. Effectuer le retrieval uniquement dans ces documents.
5. Vérifier les preuves.
6. Répondre, clarifier ou refuser.

## 9. Statuts de décision

| Statut                   | Signification                       |
| ------------------------ | ----------------------------------- |
| `answered`               | Réponse explicite et citée          |
| `clarification_required` | Information de contexte manquante   |
| `conflicting_context`    | Date et version incompatibles       |
| `no_matching_corpus`     | Aucun document ne correspond        |
| `insufficient_evidence`  | Bon corpus, mais réponse absente    |
| `conflicting_sources`    | Sources applicables contradictoires |

## 10. Principes de réponse

- répondre uniquement à partir des documents autorisés ;
- citer le document et la section utilisés ;
- ne jamais inventer une garantie ou une exclusion ;
- distinguer absence d’information et absence de couverture ;
- demander une clarification lorsqu’un filtre indispensable manque ;
- refuser lorsque les preuves sont insuffisantes ;
- ne pas utiliser une confiance auto-déclarée par le modèle.

## 11. Avertissement

Tous les produits, contrats, garanties et montants sont fictifs.

Le système constitue une démonstration technique et ne fournit aucun conseil
juridique ou assurantiel.
