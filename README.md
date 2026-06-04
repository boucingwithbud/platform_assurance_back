# Documentation — platform_assurance_back

> **Dépôt :** [github.com/paulcoffi/platform_assurance_back](https://github.com/paulcoffi/platform_assurance_back)  
> **Langage :** Python 100%  
> **Framework :** FastAPI  
> **Base de données :** PostgreSQL  
> **IA :** LangChain + Groq (LLaMA 4 Scout) + CLIP (OpenAI)

---

## Table des matières

1. [Présentation du projet](#1-présentation-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Prérequis](#3-prérequis)
4. [Structure du projet](#4-structure-du-projet)
5. [Installation et démarrage](#5-installation-et-démarrage)
6. [Configuration](#6-configuration)
7. [Pipeline d'extraction IA](#7-pipeline-dextraction-ia)
8. [Référence des routes API](#8-référence-des-routes-api)
9. [Modèles de données](#9-modèles-de-données)
10. [Schémas d'extraction par document](#10-schémas-dextraction-par-document)
11. [Dépendances](#11-dépendances)
12. [Sécurité](#12-sécurité)
13. [Dépannage](#13-dépannage)

---

## 1. Présentation du projet

**platform_assurance_back** est une API REST d'extraction intelligente de documents orientée secteur de l'assurance. Elle reçoit des images de documents officiels (JPEG/PNG) et retourne leurs données structurées en JSON, prêtes à alimenter un formulaire ou une base de données.

Le projet combine deux approches IA complémentaires :

- **Détection du type de document** : un modèle CLIP (vision-langage d'OpenAI) classe l'image pour vérifier qu'il s'agit bien du document attendu.
- **Extraction des champs** : un LLM multimodal (LLaMA 4 Scout via Groq) lit l'image et extrait les informations structurées selon un schéma Pydantic défini par document.

Les documents pris en charge sont : certificat de visite technique, facture, carte nationale d'identité (recto et verso), permis de conduire, et carte grise.

---

## 2. Architecture technique

```
┌──────────────────────────────────────────────────────────────┐
│               Client (frontend / outil tiers)                │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTP multipart/form-data
                              │ (image JPEG ou PNG)
┌─────────────────────────────▼────────────────────────────────┐
│                  FastAPI (Uvicorn)                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              main_route.py (Extraction IA)           │   │
│  │                                                      │   │
│  │  1. Validation format (JPEG/PNG)                     │   │
│  │  2. CLIP → classification du type de document        │   │
│  │  3. Encodage image → base64                          │   │
│  │  4. LLM Groq (LLaMA 4 Scout) → extraction JSON       │   │
│  │  5. Retour : { is_<type>: bool, output: {...} }       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ visite_t_     │  │ facture_      │  │ cni/permis/    │  │
│  │ postgres.py   │  │ postgres.py   │  │ carte_grise    │  │
│  │               │  │               │  │ _postgres.py   │  │
│  │ Persistance   │  │ Persistance   │  │ Persistance    │  │
│  │ BDD visite    │  │ BDD facture   │  │ BDD documents  │  │
│  └───────┬───────┘  └──────┬────────┘  └───────┬────────┘  │
└──────────┼─────────────────┼───────────────────┼────────────┘
           │                 │                   │
┌──────────▼─────────────────▼───────────────────▼────────────┐
│              PostgreSQL :5432 (assurance_db)                 │
│  visite_info · facture_info · cni_info_recto                 │
│  permis_conduire · carte_grise                               │
└──────────────────────────────────────────────────────────────┘

Modèles externes utilisés :
  ├── CLIP (openai/clip-vit-base-patch32) — local via HuggingFace/Transformers
  └── LLaMA 4 Scout 17B — via API Groq (meta-llama/llama-4-scout-17b-16e-instruct)
```

---

## 3. Prérequis

| Outil / Service | Version / Détail |
|-----------------|-----------------|
| Python | ≥ 3.10 |
| PostgreSQL | 14+ |
| Clé API Groq | Compte sur [console.groq.com](https://console.groq.com) |
| GPU (optionnel) | Recommandé pour CLIP et les modèles locaux |
| Mémoire RAM | ≥ 8 Go (16 Go recommandés pour les modèles locaux) |

> **Note CUDA :** Le `requirements.txt` inclut des packages CUDA 13 (`cuda-toolkit`, `nvidia-*`). Si vous n'avez pas de GPU NVIDIA, retirez ces dépendances lors de l'installation.

---

## 4. Structure du projet

```
platform_assurance_back/
├── routers/
│   ├── main_route.py            # Routes d'extraction IA (CLIP + LLM)
│   ├── visite_t_postgres.py     # Route persistance visite technique
│   ├── facture_postgres.py      # Route persistance facture
│   ├── cni_recto_postgres.py    # Route persistance CNI
│   ├── permis_cond_postgres.py  # Route persistance permis de conduire
│   └── carte_grise_postgres.py  # Route persistance carte grise
├── __init__.py
├── database.py                  # Connexion SQLAlchemy + session
├── img_to_b64.py                # Utilitaire image → base64
├── main.py                      # Point d'entrée FastAPI
├── models.py                    # Modèles SQLAlchemy (tables ORM)
├── requirements.txt             # Dépendances Python
├── schemas.py                   # Schémas Pydantic (validation)
└── transcrire.py                # Agents LangChain d'extraction par document
```

---

## 5. Installation et démarrage

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/paulcoffi/platform_assurance_back.git
cd platform_assurance_back
```

### Étape 2 — Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Étape 3 — Installer les dépendances

```bash
# Sans GPU (retire les dépendances CUDA)
pip install fastapi uvicorn sqlalchemy psycopg2 python-multipart pillow \
            langchain langchain-groq langchain-core pydantic python-dotenv \
            transformers torch

# Avec GPU (installation complète)
pip install -r requirements.txt
```

### Étape 4 — Configurer les variables d'environnement

Créer un fichier `.env` à la racine :

```env
GROQ_API_KEY=votre_clé_groq_ici
```

### Étape 5 — Créer la base de données PostgreSQL

```sql
CREATE DATABASE assurance_db;
CREATE USER kaydan WITH PASSWORD 'kaydan';
GRANT ALL PRIVILEGES ON DATABASE assurance_db TO kaydan;
```

### Étape 6 — Démarrer le serveur

```bash
uvicorn main:app --reload --port 8000
```

Les tables sont créées automatiquement au démarrage via `Base.metadata.create_all(engine)`.

L'API est accessible sur [http://localhost:8000](http://localhost:8000).  
La documentation interactive Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Configuration

### Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `GROQ_API_KEY` | Clé API Groq pour accéder à LLaMA 4 Scout | ✅ Oui |

### Connexion base de données (`database.py`)

```python
SQL_DB_URL = "postgresql://kaydan:kaydan@localhost:5432/assurance_db"
```

Modifier cette URL pour pointer vers votre instance PostgreSQL.

### CORS (`main.py`)

Les origines autorisées sont :

```python
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5173",
]
```

Ajouter l'URL de votre frontend en production.

### Modèle LLM (`transcrire.py`)

Le modèle utilisé est `meta-llama/llama-4-scout-17b-16e-instruct` via l'API Groq. Des alternatives commentées dans le code permettent de basculer vers :
- **Ollama** (modèles locaux) : `ChatOllama(model="qwen3-vl:4b")`
- **Google Gemini** : `ChatGoogleGenerativeAI(model="gemini-2.0-flash")`
- **HuggingFace Endpoint** : `ChatHuggingFace` avec `HuggingFaceEndpoint`

---

## 7. Pipeline d'extraction IA

Chaque endpoint d'extraction suit le même pipeline en 4 étapes :

```
Image uploadée
      │
      ▼
① Validation du format
  └─ Accepte uniquement image/jpeg et image/png
  └─ HTTPException 400 si autre format
      │
      ▼
② Classification CLIP (openai/clip-vit-base-patch32)
  └─ Chargé localement via HuggingFace Transformers
  └─ Compare l'image à des descriptions textuelles
  └─ Retourne un booléen is_<type_document>
  └─ Ex : ["N'est pas un certificat de visite technique", "Certificat de visite technique"]
      │
      ▼
③ Encodage image → base64 (img_to_b64.py)
  └─ Ouverture avec Pillow
  └─ Sauvegarde en mémoire au format PNG
  └─ Encodage base64 UTF-8
      │
      ▼
④ Extraction LLM via LangChain + Groq
  └─ Agent créé avec create_agent(model, system_prompt, response_format)
  └─ Envoi de l'image base64 + prompt au LLM multimodal
  └─ Retour structuré selon le schéma Pydantic du document
  └─ Normalisation des dates (DD-MM-YYYY → YYYY-MM-DD)
      │
      ▼
Réponse JSON :
{
  "is_<type>": true/false,
  "output": { ...champs extraits... }
}
```

### Agents LangChain disponibles (`transcrire.py`)

| Classe | Agent | Document cible |
|--------|-------|----------------|
| `TranscrireVisite` | `agent_visite_technique` | Certificat de visite technique |
| `TranscrireFacture` | `agent_facture` | Facture (manuscrite ou imprimée) |
| `TranscrireCNIRecto` | `agent_cni_recto` | CNI recto |
| `TranscrireCNIVerso` | `agent_cni_verso` | CNI verso |
| `TranscrirePermisConduire` | `agent_permis_conduire` | Permis de conduire |
| `TranscrireCarteGrise` | `agent_carte_grise` | Carte grise |

Chaque agent utilise un `system_prompt` spécialisé et un `response_format` basé sur `ToolStrategy` (structured output LangChain).

---

## 8. Référence des routes API

### Routes d'extraction IA (`main_route.py`)

Toutes ces routes acceptent un fichier image (`multipart/form-data`) et retournent le JSON extrait.

---

#### `POST /certificat_de_visite_technique`

Détecte si l'image est un certificat de visite technique et extrait ses informations.

**Entrée :** `file` (image JPEG ou PNG)

**Réponse :**
```json
{
  "is_visite_technique": true,
  "output": {
    "centre": "Centre Technique d'Abidjan",
    "immatriculation": "AB-1234-CI",
    "expiration": "2025-12-31",
    "marque": "Toyota",
    "mtt": 15000,
    ...
  }
}
```

---

#### `POST /facture_manuscrite_ou_imprime`

Détecte le type d'écriture (manuscrit / imprimé / mixte) et extrait les informations de la facture.

**Entrée :** `file` (image JPEG ou PNG)

**Réponse :**
```json
{
  "is_manuscrit": "Document imprimé",
  "output": {
    "vendeur_nom": "ENTREPRISE XYZ SARL",
    "numero_facture": "FAC-2024-001",
    "total_ttc": 118000.0,
    "devise": "XOF",
    ...
  }
}
```

Valeurs possibles pour `is_manuscrit` : `"Document manuscrit"`, `"Document imprimé"`, `"Mélange d'écritures manuscrites et imprimées"`.

---

#### `POST /cni_recto`

Détecte si l'image est une CNI et extrait les informations du recto.

**Réponse :**
```json
{
  "is_carte_didentite": true,
  "output": {
    "nom": "KONAN",
    "prenom": "Ama",
    "date_de_naissance": "1990-05-15",
    "numero_de_cni": "CI0012345678",
    ...
  }
}
```

---

#### `POST /cni_verso`

Détecte si l'image est une CNI et extrait les informations du verso (NNI, profession, autorité d'émission).

**Réponse :**
```json
{
  "is_carte_didentite": true,
  "output": {
    "nni": "1234567890123",
    "profession": "Ingénieur",
    "date_d_emission": "2020-01-10",
    "autorite_d_emission": "Ministère de l'Intérieur"
  }
}
```

---

#### `POST /permis_conduire`

Détecte si l'image est un permis de conduire et extrait ses informations.

**Réponse :**
```json
{
  "is_permis_conduire": true,
  "output": {
    "nom": "KOUASSI",
    "prenom": "Jean",
    "numero_permis": "PC-CI-0012345",
    "categories": "B, A",
    "date_expiration": "2030-06-30",
    ...
  }
}
```

---

#### `POST /carte_grise`

Détecte si l'image est une carte grise et extrait les informations du véhicule.

**Réponse :**
```json
{
  "is_carte_grise": true,
  "output": {
    "numero_immatriculation": "AB-1234-CI",
    "marque": "Toyota",
    "genre": "VP",
    "energie": "Essence",
    "puissance_fiscale": "7",
    "places_assises": "5",
    ...
  }
}
```

---

### Routes de persistance en base de données

Ces routes sauvegardent des données structurées (JSON) directement en base PostgreSQL, indépendamment du pipeline d'extraction IA. Elles permettent par exemple de persister le résultat d'une extraction après validation.

| Route | Méthode | Description |
|-------|---------|-------------|
| `POST /Visite_technique_formulaire/bdd_visite_tech` | POST | Persiste un enregistrement de visite technique |
| `POST /facture_bdd/bdd_facture` | POST | Persiste une facture |
| `POST /cni-recto-postgres/create_cni_recto_postgres` | POST | Persiste une CNI |
| `POST /Permis_conduire_formulaire/bdd_permis_conduire` | POST | Persiste un permis de conduire |
| `POST /carte_grise_formulaire/bdd_carte_grise` | POST | Persiste une carte grise |

> **Note :** La route `bdd_visite_tech` contient une validation métier : la date de visite ne peut pas être dans le futur ni antérieure à la date de mise en circulation du véhicule.

---

## 9. Modèles de données

### Table `visite_info`

Données d'un certificat de visite technique et vignette.

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | Integer PK | |
| `centre` | String | Centre de contrôle |
| `mtt` | Integer | Montant de la taxe (sans unité) |
| `cat` | String | Catégorie du véhicule |
| `kms` | String | Kilométrage |
| `stat` | String | Statut (admis/refusé) |
| `ville` | String | Ville du centre |
| `immatriculation` | String | Numéro de plaque |
| `expiration` | Date | Date d'expiration |
| `marque` | String | Marque du véhicule |
| `type` | String | Type de véhicule |
| `numero_serie` | String | Numéro de série |
| `puis_fis_cv` | Integer | Puissance fiscale en CV |
| `mise_en_circulation` | Date | Date de première mise en circulation |
| `observations` | String | Observations du contrôleur |
| `quotite` | Integer | Quotité (valeur numérique sans unité) |
| `numero_vignette` | String | Numéro de vignette |
| `ncc` | String | NCC |
| `responsable` | String | Responsable du centre |

---

### Table `facture_info`

Données complètes d'une facture commerciale.

| Groupe | Colonnes |
|--------|---------|
| Émetteur | `vendeur_nom`, `vendeur_adresse`, `vendeur_telephone`, `vendeur_email`, `vendeur_rccm`, `vendeur_compte_contribuable` |
| Client | `client_nom`, `client_adresse`, `client_telephone`, `client_email`, `client_code` |
| Identifiants | `numero_facture`, `date_facture`, `date_echeance`, `numero_bon_commande`, `numero_bon_livraison`, `objet` |
| Lignes (JSON) | `lignes_descriptions`, `lignes_references`, `lignes_quantites`, `lignes_unites`, `lignes_prix_unitaires`, `lignes_taux_tva`, `lignes_montants_ht`, `lignes_montants_ttc` |
| Totaux | `total_ht`, `remise`, `total_ht_apres_remise`, `montant_tva`, `taux_tva_global`, `total_ttc`, `acompte`, `reste_a_payer`, `devise` |
| Paiement | `mode_paiement`, `banque`, `iban_rib`, `numero_cheque` |
| Divers | `mentions_legales`, `notes`, `cachet_signature` |

Les colonnes de lignes de détail sont stockées en **JSON** (tableaux de valeurs parallèles), ce qui permet de gérer un nombre variable de lignes par facture.

---

### Table `cni_info_recto`

| Colonne | Type | Description |
|---------|------|-------------|
| `nom`, `prenom` | String | Identité |
| `date_de_naissance` | Date | |
| `lieu_de_naissance` | String | |
| `sexe` | String | M ou F |
| `taille` | Float | En cm |
| `date_d_expiration` | Date | |
| `numero_de_cni` | String | |
| `nationalite` | String | |
| `nni` | String | Numéro National d'Identification (verso) |
| `profession` | String | |
| `date_d_emission` | String | |
| `autorite_d_emission` | String | |

---

### Table `permis_conduire`

| Colonne | Type | Description |
|---------|------|-------------|
| `nom`, `prenom` | String | |
| `date_naissance`, `lieu_naissance` | Date/String | |
| `adresse` | String | |
| `lieu_delivrance` | String | |
| `date_expiration` | Date | |
| `numero_permis` | String | |
| `categories` | String | Ex : "A, B, C" |

---

### Table `carte_grise`

| Colonne | Type | Description |
|---------|------|-------------|
| `numero_immatriculation` | String (index) | |
| `numero_carte_grise` | String (unique) | |
| `date_premiere_mise_circulation` | Date | |
| `date_edition_carte_grise` | Date | |
| `identite_titulaire` | String | |
| `marque`, `genre`, `type_commercial` | String | |
| `couleur`, `carrosserie`, `energie` | String | |
| `usage_vehicule` | String | |
| `nombre_essieux`, `places_assises` | Integer | |
| `puissance_fiscale`, `cylindree_cc` | Integer | |
| `masse_vehicule` (PTAC), `pv`, `cu` | Integer | En kg |

---

## 10. Schémas d'extraction par document

Les schémas Pydantic dans `schemas.py` définissent la structure de sortie de chaque type de document. Chaque champ est accompagné d'une `description` qui guide le LLM pour localiser l'information dans l'image.

### Exemple — Visite technique

```python
centre: Optional[str] = Field(description="à droite de CENTRE dans le document")
immatriculation: Optional[str] = Field(description="Juste en dessous de Immatriculation")
mtt: Optional[int] = Field(description="à droite de MTT dans le document")
```

### Exemple — Facture

Les lignes de détail sont extraites en **listes parallèles** :
```python
lignes_descriptions: Optional[List[str]]   # ["Article A", "Article B"]
lignes_quantites:    Optional[List[int]]    # [2, 5]
lignes_prix_unitaires: Optional[List[float]] # [10000.0, 2500.0]
```

### Normalisation des dates

Le LLM est instruit dans chaque prompt de convertir les dates du format `DD-MM-YYYY` (format courant sur les documents ivoiriens) vers `YYYY-MM-DD` (format ISO, compatible avec Pydantic `date`).

---

## 11. Dépendances

### Bibliothèques clés

| Bibliothèque | Usage |
|--------------|-------|
| `fastapi` | Framework REST |
| `uvicorn` | Serveur ASGI |
| `sqlalchemy` + `psycopg2` | ORM et driver PostgreSQL |
| `pydantic` | Validation et schémas structurés |
| `python-multipart` | Upload de fichiers |
| `pillow` | Manipulation d'images (ouverture, encodage) |
| `langchain` + `langchain-groq` | Orchestration LLM et agents |
| `langchain-core` | Primitives LangChain (messages, outils) |
| `groq` | Client API Groq |
| `transformers` | Modèle CLIP via HuggingFace |
| `torch` | Backend PyTorch pour CLIP |
| `python-dotenv` | Chargement des variables d'environnement |

### Modèles IA

| Modèle | Rôle | Exécution |
|--------|------|-----------|
| `openai/clip-vit-base-patch32` | Classification du type de document | Local (via `transformers`) |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Extraction multimodale des champs | API distante (Groq) |

---

## 12. Sécurité

**Points d'attention :**

- La clé `GROQ_API_KEY` ne doit jamais être committée. Toujours utiliser un `.env` ajouté au `.gitignore`.
- Les identifiants PostgreSQL (`kaydan:kaydan`) sont en clair dans `database.py` — les externaliser en variable d'environnement pour la production.
- L'API n'implémente pas d'authentification. En production, ajouter un middleware d'authentification (JWT, API key, OAuth2) avant exposition publique.
- Le format des fichiers est validé par `content_type` (JPEG/PNG), mais une validation supplémentaire du contenu binaire est recommandée en production.

---

## 13. Dépannage

### `GROQ_API_KEY` manquante

```
Error: GROQ_API_KEY not found
```

Vérifier que le fichier `.env` existe à la racine et contient la clé.

### Erreur de connexion PostgreSQL

```
sqlalchemy.exc.OperationalError: could not connect to server
```

Vérifier que PostgreSQL est démarré, que la base `assurance_db` existe, et que l'utilisateur `kaydan` a les droits. Adapter `SQL_DB_URL` dans `database.py` si nécessaire.

### `400 — Fichier non supporté`

L'endpoint n'accepte que les images JPEG (`image/jpeg`) et PNG (`image/png`). Les PDF, TIFF ou autres formats doivent être convertis au préalable.

### CLIP charge lentement au premier démarrage

Le modèle `clip-vit-base-patch32` est téléchargé depuis HuggingFace Hub au premier lancement (~600 Mo). Les démarrages suivants utilisent le cache local.

### Le LLM retourne des champs `null`

Si certains champs sont absents ou illisibles dans l'image, le LLM retourne `null` pour ces champs (comportement attendu car tous les champs sont `Optional`). Pour améliorer la qualité d'extraction, s'assurer que l'image est nette, bien éclairée, et non tronquée.

### Erreur sur la route `bdd_visite_tech`

```json
{ "detail": "La date de visite ne peut pas être dans le futur." }
```

Ce contrôle métier est intentionnel dans `visite_t_postgres.py`. La `date_visite` fournie doit être ≤ à la date du jour et ≥ à la `date_mise_circulation`.

---

*Documentation générée à partir du code source du dépôt [paulcoffi/platform_assurance_back](https://github.com/paulcoffi/platform_assurance_back).*
