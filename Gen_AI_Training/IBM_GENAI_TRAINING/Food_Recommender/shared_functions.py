"""Shared retrieval helpers for the Food Recommender demo.

This module is the retrieval half of a deliberately small RAG
(retrieval-augmented generation) pipeline: it turns the static JSON food
catalogue into a vector index and answers natural-language food queries against
that index.  `interactive_search.py` imports it with
``from shared_functions import *`` and owns the conversational/CLI layer; every
piece of data handling and vector-store access lives here so the two stays in
sync.

Pipeline
--------
::

    FoodDataSet.json
          |
          |  load_food_data()                       (parse + normalise)
          v
      List[Dict] -------------------.
          |                         |
          |  create_similarity_search_collection()
          v                         |
      empty ChromaDB collection <----'
          |
          |  populate_similarity_collection()       (embed + insert)
          v
      populated collection
          |
          |  perform_similarity_search()            (semantic query)
          |  perform_filtered_similarity_search()   (semantic + metadata)
          |  perform_calorie_aware_search()         (intent parse + re-rank,
          |                                          what the CLI actually calls)
          v
      List[Dict] of ranked results

Design notes
------------
* **Ephemeral store.** ``chromadb.Client()`` is the in-memory client, so the
  index is rebuilt from scratch on every process start and nothing is written to
  disk.  That keeps the demo reproducible, but it also means a collection must
  be populated before any query is issued.
* **Local embeddings.** Vectors come from the ``all-MiniLM-L6-v2`` sentence
  transformer, so querying needs no API key and no per-call quota (the model
  weights are downloaded once and then cached on disk by
  ``sentence-transformers``).
* **Fail-soft contract.** Apart from :func:`create_similarity_search_collection`,
  every public helper catches its own exceptions, prints a diagnostic and
  returns an empty list instead of raising.  Callers can therefore always
  iterate over the result without wrapping it in ``try``/``except``.
* **Distance metric lives in two places (0.6.x).** On chromadb 0.6.0 the
  ``configuration`` object only *records* ``hnsw.space`` -- the local HNSW
  segment is built from the collection *metadata* instead (``HnswParams`` reads
  ``hnsw:space`` and defaults to ``l2``).  If the metric is not set in both
  places the index silently uses squared-L2, and since the embeddings are
  unit-normalised that turns ``1 - distance`` into ``2*cos - 1``: correct
  ranking, wrong (even negative) percentages.  See
  :func:`create_similarity_search_collection`.
* **ChromaDB version compatibility.** ``requirements.txt`` pins ``chromadb``
  0.6.0, which takes *typed* configuration objects when a collection is created;
  the 1.x line takes a plain mapping instead.  Building the collection
  configuration is therefore isolated in
  :func:`_build_collection_configuration`, which prefers the 0.6.x shape and
  falls back to the 1.x shape.  Passing the wrong one is what produces the
  ``'dict' object has no attribute 'to_json'`` failure at start-up.
* **Single source of truth for the record shape.** :func:`load_food_data`
  normalises every record up front, so the rest of the module may assume a fixed
  set of keys.  Raw records in ``FoodDataSet.json`` look like::

      {
        "food_id": 1,
        "food_name": "Apple Pie",
        "food_description": "A classic dessert made with ...",
        "food_calories_per_serving": 320,
        "food_nutritional_factors": {
            "carbohydrates": "42g", "protein": "2g", "fat": "16g"
        },
        "food_ingredients": ["Apples", "Flour", "Butter", "Sugar"],
        "food_health_benefits": "Rich in antioxidants and dietary fiber",
        "cooking_method": "Baking",
        "cuisine_type": "American",
        "food_features": {
            "taste": "sweet", "texture": "crisp and tender",
            "appearance": "golden brown", "preparation": "baked",
            "serving_type": "hot"
        }
      }

  :func:`load_food_data` adds a flattened ``taste_profile`` string to each
  record and guarantees the fields consumed by the embedding and metadata
  builders always exist.
"""

import chromadb
from chromadb.utils import embedding_functions
import json
# ``re`` and ``numpy`` are not used directly here; they stay imported so that
# callers doing ``from shared_functions import *`` keep receiving them as they
# historically did (removing them would be a silent breaking change).
import re
import numpy as np
# ``Any`` / ``Optional`` are kept for the same reason: they are part of this
# module's public typing imports.
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Vector-store client
# ---------------------------------------------------------------------------
# Module-level in-memory client shared by every collection created below.  Use
# ``chromadb.PersistentClient(path=...)`` instead if the index should survive a
# restart; nothing else in this module depends on which flavour is used.
client = chromadb.Client()


# ---------------------------------------------------------------------------
# Stage 1 - data loading and normalisation
# ---------------------------------------------------------------------------
def load_food_data(file_path: str) -> List[Dict]:
    """Load the food catalogue from ``file_path`` and normalise every record.

    The dataset is hand-authored JSON, so field presence and types are not
    guaranteed across records.  Rather than validating strictly and failing, this
    function *repairs* each record so downstream code (embedding-text builder,
    metadata builder, filters) never needs defensive lookups.

    Args:
        file_path: Path to a JSON file holding a list of food objects.  The
            expected -- but not enforced -- schema is documented in the module
            docstring.

    Returns:
        The list of normalised food dictionaries, in file order.  Returns ``[]``
        (never raises) when the file is missing, unreadable or not valid JSON,
        so the caller can print a friendly message instead of crashing.

    Normalisation performed on each record
    --------------------------------------
    1. ``food_id`` is coerced to ``str`` and auto-generated as ``"1"``, ``"2"``,
       ... when missing.  ChromaDB document IDs must be strings, so normalising
       here keeps :func:`populate_similarity_collection` simple.
    2. Missing optional fields get inert defaults:
       ``food_ingredients -> []``, ``food_description -> ""``,
       ``cuisine_type -> "Unknown"``, ``food_calories_per_serving -> 0``.
       ``food_name`` is deliberately *not* defaulted: a nameless dish cannot be
       shown to the user and ChromaDB metadata rejects ``None`` values anyway.
    3. ``taste_profile`` is derived by flattening the nested ``food_features``
       dict into a comma-separated string, e.g. ``"sweet, crisp and tender,
       golden brown"``.  Flattening happens here so the embedding builder has one
       predictable string field to work with instead of a dict whose key set
       differs per dish.  Truthy values are kept; empty/``None`` features are
       dropped.

    Side effects:
        Prints ``Successfully loaded N food items from <path>`` on success, or
        ``Error loading food data: <reason>`` on failure.

    Example:
        >>> foods = load_food_data("./FoodDataSet.json")
        >>> "taste_profile" in foods[0]
        True
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)

        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])

            # Ensure required fields exist
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0

            # Extract taste features from nested food_features if available.
            # The dict is flattened into one string because both the embedded
            # document and the ChromaDB metadata are flat values.
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                item['taste_profile'] = ', '.join(taste_features)
            else:
                item['taste_profile'] = ''

        print(f"Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data

    except Exception as e:
        print(f"Error loading food data: {e}")
        return []


# ---------------------------------------------------------------------------
# Stage 2 - collection creation
# ---------------------------------------------------------------------------
def _build_collection_configuration(hnsw_space: str, embedding_function):
    """Build a ``create_collection(configuration=...)`` value ChromaDB accepts.

    ChromaDB changed the *type* of this argument between releases, and the two
    shapes are not interchangeable:

    * **0.6.x** (the version pinned in ``requirements.txt``) expects typed
      configuration objects and serialises them with ``configuration.to_json()``.
      Handing it the 1.x mapping raises
      ``AttributeError: 'dict' object has no attribute 'to_json'`` -- the error
      that ``interactive_search.main()`` reports as
      ``Error initializing system: ...``.
    * **1.x** expects a ``CreateCollectionConfiguration`` mapping such as
      ``{"hnsw": {"space": ...}, "embedding_function": ...}``.

    The typed objects are therefore tried first and the mapping is used as a
    fallback, so one module works against both layouts.  Only the public
    ``hnsw_space`` knob is exposed; every other HNSW parameter keeps its ChromaDB
    default.

    Args:
        hnsw_space: Distance metric for the HNSW index. ``"cosine"`` here, so the
            ``1 - distance`` conversion in the search helpers is meaningful.
        embedding_function: The sentence-transformer embedding function; only the
            1.x mapping form carries it inside the configuration.

    Returns:
        ``(configuration, needs_embedding_function_kwarg,
        needs_legacy_hnsw_metadata)``.

        * ``needs_embedding_function_kwarg`` -- ``True`` on 0.6.x, where the
          embedding function is a first-class ``create_collection`` argument;
          ``False`` on 1.x, where it travels inside the configuration mapping.
        * ``needs_legacy_hnsw_metadata`` -- ``True`` on 0.6.x, because that
          release's local HNSW segment *ignores* this configuration object and
          reads ``hnsw:space`` from the collection metadata instead.  The caller
          must therefore also merge ``{"hnsw:space": hnsw_space}`` into the
          metadata, otherwise the index falls back to ``l2``.  1.x rejects
          ``hnsw:*`` metadata keys, hence ``False`` there.
    """
    try:
        # chromadb >= 0.6.0: typed, serialisable configuration objects. The
        # *Internal* classes are used deliberately: the public interface classes
        # serialise their own class name into `_type`, and ChromaDB later tries
        # to rebuild the object as the Internal type, raising
        # "Trying to instantiate configuration of type ...Internal from JSON with
        # type ...Interface". Building the Internal objects directly keeps the
        # serialise/deserialise round trip consistent.
        from chromadb.api.configuration import (
            CollectionConfigurationInternal,
            ConfigurationParameter,
            HNSWConfigurationInternal,
        )

        configuration = CollectionConfigurationInternal(
            parameters=[
                ConfigurationParameter(
                    name="hnsw_configuration",
                    value=HNSWConfigurationInternal(
                        parameters=[
                            ConfigurationParameter(name="space", value=hnsw_space)
                        ]
                    ),
                )
            ]
        )
        return configuration, True, True
    except (ImportError, TypeError):
        # 1.x layout: everything travels in a single mapping.
        return {
            "hnsw": {"space": hnsw_space},
            "embedding_function": embedding_function,
        }, False, False


def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    """Create -- or recreate -- an embedding-backed ChromaDB collection.

    The function is destructive on purpose: any existing collection with the same
    name is dropped first.  That makes repeated demo runs idempotent, so stale
    rows from an older version of the JSON file can never be mixed with fresh
    embeddings.

    Args:
        collection_name: Unique name of the collection within the in-memory
            client.
        collection_metadata: Optional metadata stored with the collection, e.g.
            ``{"description": "A collection for interactive food search"}``.
            ChromaDB only accepts scalar (str/int/float/bool) values here.

    Returns:
        A fresh, empty collection whose embedding function and index settings are
        already configured, ready to be passed to
        :func:`populate_similarity_collection`.

    Embedding and index configuration
    ---------------------------------
    * ``SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")``
      produces 384-dimensional vectors locally on CPU: small, fast and
      deterministic across runs, which is plenty for a catalogue of a few hundred
      dishes.
    * ``hnsw.space = "cosine"`` makes the HNSW index rank by cosine distance.
      Cosine distance spans ``[0, 2]``: for the normalised MiniLM embeddings used
      here a match approaches ``0`` and unrelated dishes sit near ``1``.  That is
      what makes the ``similarity = 1 - distance`` conversion in the search
      helpers read as a cosine similarity (which can dip slightly below zero for
      unrelated pairs).  Setting the metric is a two-step dance on chromadb
      0.6.x: the configuration object above records it, *and* the legacy
      ``hnsw:space`` metadata key must be present because that is the only thing
      the local HNSW segment reads.  Skip the metadata and ChromaDB quietly
      indexes with squared-L2, which pushes ``1 - distance`` negative for any
      match with cosine similarity below ``0.5``.
    * The embedding function is bound to the *collection* rather than passed per
      call, so ingestion and query code never has to know which model is in use
      -- and queries are automatically embedded with exactly the same model that
      indexed the documents.  Where it is bound depends on the installed ChromaDB
      version; see :func:`_build_collection_configuration`.

    Note:
        The ``try``/``except`` around ``delete_collection`` is intentionally
        broad.  A brand-new client has nothing to delete and ChromaDB raises for
        that case; there is nothing useful to do about it.
    """
    try:
        # Try to delete existing collection to start fresh
        client.delete_collection(collection_name)
    except:
        pass

    # Create embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Select the cosine metric for the HNSW index (and, on 1.x, the embedding
    # function) using whichever shape the installed ChromaDB expects.
    (
        configuration,
        needs_embedding_function_kwarg,
        needs_legacy_hnsw_metadata,
    ) = _build_collection_configuration("cosine", sentence_transformer_ef)

    # chromadb 0.6.x builds its HNSW segment from the collection metadata rather
    # than from the configuration object, so the legacy `hnsw:space` key is what
    # actually selects cosine distance. Without it the index would use squared-L2
    # and `1 - distance` would stop being a cosine similarity.
    metadata = dict(collection_metadata or {})
    if needs_legacy_hnsw_metadata:
        metadata["hnsw:space"] = "cosine"

    create_kwargs = dict(
        name=collection_name,
        metadata=metadata or None,
        configuration=configuration,
    )
    if needs_embedding_function_kwarg:
        # chromadb 0.6.x: the embedding function is a first-class argument
        # instead of part of the configuration mapping.
        create_kwargs["embedding_function"] = sentence_transformer_ef

    return client.create_collection(**create_kwargs)


# ---------------------------------------------------------------------------
# Stage 3 - ingestion (embedding + metadata)
# ---------------------------------------------------------------------------
def populate_similarity_collection(collection, food_items: List[Dict]):
    """Embed ``food_items`` and insert them into ``collection``.

    Three parallel lists are accumulated and handed to ``collection.add()`` in a
    single batch.  Their order matters: ``documents[i]`` is the text that gets
    embedded for ``ids[i]``, and ``metadatas[i]`` is the payload that filters and
    result formatting read back later.  There is no explicit embedding call --
    the collection's embedding function runs implicitly inside ``add()``.

    Args:
        collection: A collection returned by
            :func:`create_similarity_search_collection`.
        food_items: Normalised records produced by :func:`load_food_data`.

    Returns:
        ``None``; the function exists for its side effect on ``collection``.

    What gets embedded
    ------------------
    Each record is flattened into one "document" that concatenates name,
    description, ingredients, cuisine, cooking method, taste features, health
    benefits and nutritional factors.  Embedding one rich document (instead of,
    say, only the dish name) is what lets a free-text query such as *"something
    sweet and baked with apples"* land near Apple Pie: every descriptive field
    contributes signal to the same vector.  Fields are only appended when
    present, so short records still produce a valid document.

    Metadata stored next to each vector
    -----------------------------------
    ``name``, ``cuisine_type``, ``ingredients``, ``calories``, ``description``,
    ``cooking_method``, ``health_benefits``, ``taste_profile``.  Two deliberate
    choices: ``calories`` stays numeric so ``$lte`` range filters work, and
    ``ingredients`` is joined into a string because ChromaDB metadata cannot hold
    lists.  This duplicates data that is also inside the embedded document, but
    it is the only thing available when building result dictionaries, since the
    document text itself is not returned by :func:`perform_similarity_search`.

    Duplicate handling
    ------------------
    Document IDs must be unique per collection.  If the JSON repeats a
    ``food_id``, later records are suffixed ``<id>_1``, ``<id>_2``, ... so the
    batch insert cannot fail on a duplicate-ID error.
    """
    documents = []
    metadatas = []
    ids = []

    # Create unique IDs to avoid duplicates
    used_ids = set()

    for i, food in enumerate(food_items):
        # Create comprehensive text for embedding using rich JSON structure.
        # Every sentence fragment is optional except the name, which
        # load_food_data leaves intact and which ChromaDB requires as metadata.
        text = f"Name: {food['food_name']}. "
        text += f"Description: {food.get('food_description', '')}. "
        text += f"Ingredients: {', '.join(food.get('food_ingredients', []))}. "
        text += f"Cuisine: {food.get('cuisine_type', 'Unknown')}. "
        text += f"Cooking method: {food.get('cooking_method', '')}. "

        # Add taste profile from food_features
        taste_profile = food.get('taste_profile', '')
        if taste_profile:
            text += f"Taste and features: {taste_profile}. "

        # Add health benefits if available
        health_benefits = food.get('food_health_benefits', '')
        if health_benefits:
            text += f"Health benefits: {health_benefits}. "

        # Add nutritional information. The factors dict is rendered as
        # "key: value" pairs so amounts ("42g") survive embedding and can still
        # be read by a model answering from retrieved context.
        if 'food_nutritional_factors' in food:
            nutrition = food['food_nutritional_factors']
            if isinstance(nutrition, dict):
                nutrition_text = ', '.join([f"{k}: {v}" for k, v in nutrition.items()])
                text += f"Nutrition: {nutrition_text}."

        # Generate unique ID to avoid duplicates: fall back to the loop index
        # when food_id is absent, then suffix collisions with _1, _2, ...
        base_id = str(food.get('food_id', i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)

        documents.append(text)
        ids.append(unique_id)
        metadatas.append({
            "name": food["food_name"],
            "cuisine_type": food.get("cuisine_type", "Unknown"),
            "ingredients": ", ".join(food.get("food_ingredients", [])),
            "calories": food.get("food_calories_per_serving", 0),
            "description": food.get("food_description", ""),
            "cooking_method": food.get("cooking_method", ""),
            "health_benefits": food.get("food_health_benefits", ""),
            "taste_profile": food.get("taste_profile", "")
        })

    # Add all data to collection; the embedding function bound to the collection
    # converts `documents` into vectors in this single call.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Added {len(food_items)} food items to collection")


# ---------------------------------------------------------------------------
# Stage 4 - similarity search
# ---------------------------------------------------------------------------
def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Run a pure semantic query against the collection.

    Args:
        collection: A populated ChromaDB collection.
        query: Natural-language query, e.g. ``"spicy vegetarian dinner"``.  It is
            embedded with the collection's embedding function -- the same model
            used to index the documents -- before the nearest-neighbour lookup.
        n_results: Maximum number of neighbours to return.  The default is
            chosen here (5) rather than relying on ChromaDB's default of 10,
            because the recommender only shows a handful of options at a time.

    Returns:
        Up to ``n_results`` dictionaries ordered from most to least similar::

            {
                "food_id": "1",                     # ChromaDB document id
                "food_name": "Apple Pie",
                "food_description": "...",
                "cuisine_type": "American",
                "food_calories_per_serving": 320,
                "similarity_score": 0.72,           # 1 - cosine distance
                "distance": 0.28,                   # raw cosine distance
            }

        An empty list when the query returns no hits or when anything fails.

    Interpretation:
        ``similarity_score`` is literally ``1 - cosine distance``, i.e. the cosine
        similarity between the query and the dish embedding.  It therefore spans
        ``[-1, 1]``; with this model relevant dishes land near ``1`` and unrelated
        ones near or slightly below ``0``, so ``interactive_search`` can render it
        as a percentage.  Treat it as a *relative* ranking signal, not a
        calibrated probability, and avoid thresholding "relevance" on it blindly.

    Result indexing:
        ChromaDB always returns per-query lists, even for a single query, hence
        the ``results['ids'][0][i]`` shape: ``[0]`` selects the only query, ``[i]``
        selects the hit.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Defensive guard: an empty/failed query may yield no ids at all.
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []

        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance). Valid because the HNSW
            # index is really cosine: on 0.6.x the metric has to be set in the
            # collection metadata, not just in the configuration object (see
            # create_similarity_search_collection).
            similarity_score = 1 - results['distances'][0][i]

            result = {
                'food_id': results['ids'][0][i],
                'food_name': results['metadatas'][0][i]['name'],
                'food_description': results['metadatas'][0][i]['description'],
                'cuisine_type': results['metadatas'][0][i]['cuisine_type'],
                'food_calories_per_serving': results['metadatas'][0][i]['calories'],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)

        return formatted_results

    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []


def perform_filtered_similarity_search(collection, query: str, cuisine_filter: str = None,
                                     max_calories: int = None, n_results: int = 5) -> List[Dict]:
    """Semantic query restricted by exact-match and range metadata constraints.

    Ranking and result shape are identical to :func:`perform_similarity_search`;
    the difference is that the candidate set is filtered *inside* ChromaDB using
    the metadata written by :func:`populate_similarity_collection`, so the
    ``n_results`` nearest neighbours are drawn from the filtered subset rather
    than from the whole catalogue.

    Args:
        collection: A populated ChromaDB collection.
        query: Natural-language query (see :func:`perform_similarity_search`).
        cuisine_filter: Exact, case-sensitive match on the ``cuisine_type``
            metadata field.  ``"American"`` matches, ``"american"`` does not.
        max_calories: Inclusive upper bound (ChromaDB ``$lte``) applied to the
            numeric ``calories`` metadata.  ``None`` or ``0`` disables the
            constraint, so a literal zero-calorie limit cannot be expressed.
        n_results: Maximum number of neighbours to return.

    Returns:
        A ranked list of result dicts -- the same schema as
        :func:`perform_similarity_search` -- that may be *shorter* than
        ``n_results`` when the filter leaves fewer matches.  An over-restrictive
        filter (or any exception) produces ``[]``.

    Filter construction:
        Each active constraint becomes one ChromaDB ``where`` operand.  A single
        constraint is passed through as-is; several are combined under ``$and``.
        With no constraints ``where_clause`` stays ``None``, in which case the
        call degenerates to :func:`perform_similarity_search`.
    """
    where_clause = None

    # Build filters list: one operand per supplied constraint.
    filters = []
    if cuisine_filter:
        filters.append({"cuisine_type": cuisine_filter})

    # `calories` metadata is numeric, so range operators such as $lte are valid.
    if max_calories:
        filters.append({"calories": {"$lte": max_calories}})

    # Construct where clause based on number of filters
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        # Same defensive guard as the unfiltered search: filters can legitimately
        # match nothing.
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []

        formatted_results = []
        for i in range(len(results['ids'][0])):
            similarity_score = 1 - results['distances'][0][i]

            result = {
                'food_id': results['ids'][0][i],
                'food_name': results['metadatas'][0][i]['name'],
                'food_description': results['metadatas'][0][i]['description'],
                'cuisine_type': results['metadatas'][0][i]['cuisine_type'],
                'food_calories_per_serving': results['metadatas'][0][i]['calories'],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)

        return formatted_results
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return []


# ---------------------------------------------------------------------------
# Stage 5 - intent-aware search (what the CLI actually calls)
# ---------------------------------------------------------------------------
# Common misspellings seen in free-text queries.  The embedding model is
# word-piece based, so an out-of-vocabulary token such as "calroes" contributes
# almost no signal to the query vector; repairing it before embedding keeps the
# retrieval on topic.
_QUERY_TYPO_FIXES = {
    "deserts": "desserts",
    "desert": "dessert",
    "desset": "dessert",
    "calroes": "calories",
    "caloires": "calories",
    "calries": "calories",
    "calory": "calories",
    "protien": "protein",
    "vegitarian": "vegetarian",
    "italain": "italian",
    "mexcian": "mexican",
}

# "lowest calories", "least calorie", "fewest kcal", ... -> order by calories.
_LOWEST_CALORIE_RE = re.compile(
    r"\b(?:lowest|least|fewest|minimum|minimal|lightest)\b[\s\w]*"
    r"\b(?:calorie|calories|cal|kcal)\b",
    re.IGNORECASE,
)

# "low calorie", "low-calorie", "lower calories", "healthy calories" -> order.
_LOW_CALORIE_RE = re.compile(
    r"\b(?:low|lower|light|healthy)\b[\s-]*(?:calorie|calories|cal|kcal)\b",
    re.IGNORECASE,
)

# "under 300 calories", "less than 250 kcal", "max 200 cal", "< 150" -> hard cap.
_MAX_CALORIE_RE = re.compile(
    r"\b(?:under|below|less than|at most|max|maximum|up to|upto)\s*"
    r"(\d{1,4})\s*(?:kcal|calories|cal)?\b"
    r"|<\s*(\d{1,4})",
    re.IGNORECASE,
)


def _known_cuisines(collection) -> List[str]:
    """Return the distinct ``cuisine_type`` values stored in ``collection``.

    Used to recognise cuisine names in free text (``"thai food"`` ->
    ``cuisine_filter="Thai"``) without hard-coding a list that would drift from
    ``FoodDataSet.json``.  The whole metadata payload is read once per query,
    which is fine at demo scale (a few hundred rows, in-memory) but would need to
    be cached or replaced by a facet lookup for a large catalogue.

    Args:
        collection: A populated ChromaDB collection.

    Returns:
        Sorted list of cuisine names, or ``[]`` when the collection is empty or
        the read fails.  Every failure is swallowed: cuisine detection is a
        convenience, never a requirement.
    """
    try:
        metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
        cuisines = {
            metadata.get("cuisine_type")
            for metadata in metadatas
            if isinstance(metadata, dict) and metadata.get("cuisine_type")
        }
        return sorted(cuisines)
    except Exception:
        return []


def parse_search_constraints(query: str,
                            known_cuisines: Optional[List[str]] = None) -> Dict[str, Any]:
    """Turn a free-text query into retrieval constraints plus a clean query.

    Vector search alone cannot honour words like *"lowest"* or *"under 300"*:
    the query is embedded as text and calories never take part in the ranking.
    This parser recovers the numeric intent with a few explicit regexes so the
    behaviour stays predictable and explainable (an LLM would be overkill for a
    demo and would add latency, cost and non-determinism).

    Recognised intents
    ------------------
    * ``"lowest calories"``, ``"least calorie"``, ``"fewest kcal"``,
      ``"low calorie"``, ``"lower calories"`` -> ``sort_by_calories=True``.
      The results are *re-ordered*, not filtered, so the user still sees
      relevant dishes.
    * ``"under 300 calories"``, ``"less than 250 kcal"``, ``"max 200 cal"``,
      ``"< 150"`` -> ``max_calories=N`` (an inclusive upper bound, applied by
      ChromaDB as ``$lte``).
    * A cuisine name that exists in the collection (e.g. ``"thai"``) ->
      ``cuisine_filter`` with the canonical spelling.

    Args:
        query: Raw user input.
        known_cuisines: Cuisine names to look for, normally from
            :func:`_known_cuisines`.  Defaults to an empty list, which disables
            cuisine detection.

    Returns:
        A dictionary with four keys::

            {
                "max_calories": int | None,      # hard upper bound
                "sort_by_calories": bool,        # order results ascending
                "cuisine_filter": str | None,    # canonical cuisine name
                "semantic_query": str,           # typo-repaired query text
            }

    Example:
        >>> parse_search_constraints("deserts with lowest calroes")
        {'max_calories': None, 'sort_by_calories': True,
         'cuisine_filter': None, 'semantic_query': 'desserts with lowest calories'}
    """
    # Repair known misspellings without touching the rest of the query, so the
    # user's wording (and any cuisine name) still reaches the embedding model.
    semantic_query = re.sub(
        r"[A-Za-z]+",
        lambda match: _QUERY_TYPO_FIXES.get(match.group(0).lower(), match.group(0)),
        query,
    )

    max_calories = None
    max_match = _MAX_CALORIE_RE.search(semantic_query)
    if max_match:
        # The pattern has two alternatives: "<word> 300" or "< 300".
        max_calories = int(max_match.group(1) or max_match.group(2))

    sort_by_calories = bool(
        _LOWEST_CALORIE_RE.search(semantic_query) or _LOW_CALORIE_RE.search(semantic_query)
    )

    # Match cuisines on word boundaries so "thai" does not fire inside "thailand"
    # style words by accident, and return the canonical spelling ChromaDB stores.
    cuisine_filter = None
    for cuisine in known_cuisines or []:
        pattern = r"\b" + re.escape(cuisine.lower()) + r"\b"
        if re.search(pattern, semantic_query.lower()):
            cuisine_filter = cuisine
            break

    return {
        "max_calories": max_calories,
        "sort_by_calories": sort_by_calories,
        "cuisine_filter": cuisine_filter,
        "semantic_query": semantic_query,
    }


def perform_calorie_aware_search(collection, query: str, n_results: int = 5,
                                 candidate_pool: int = 50) -> List[Dict]:
    """Answer a user query, honouring calorie intent when it is present.

    This is the function the CLI should call.  It behaves exactly like
    :func:`perform_similarity_search` for ordinary queries and only adds work
    when :func:`parse_search_constraints` finds a constraint.

    Strategy -- relevance first, numbers second
    -------------------------------------------
    1. Parse the query.  If nothing was recognised, delegate straight to
       :func:`perform_similarity_search` (no behaviour change, no extra cost).
    2. Retrieve a wider *semantic* candidate pool (``candidate_pool`` hits) using
       :func:`perform_filtered_similarity_search`, so cuisine/calorie limits are
       applied inside ChromaDB.
    3. If the user asked for the *lowest* calories, sort **that candidate pool**
       by calories ascending and keep the best ``n_results``.

    Ordering by calories across the whole catalogue would be wrong: the dataset's
    absolute lowest items are condiments (Vinegar 3 kcal, Pickles 5 kcal), which
    answer "lowest calories" but not "lowest-calorie dessert".  Sorting the
    semantically relevant pool keeps the result both on topic and correctly
    ordered.

    Args:
        collection: A populated ChromaDB collection.
        query: Raw user input, constraints and all.
        n_results: How many recommendations to return.
        candidate_pool: How many semantic neighbours to consider before
            re-ordering.  Must be at least ``n_results``; larger values trade a
            little latency for a better chance that truly low-calorie dishes
            appear in the pool.  The default (50) is roughly a quarter of the
            demo catalogue -- wide enough to reach low-calorie desserts, narrow
            enough to keep condiments such as Vinegar (3 kcal) out of a
            dessert query.

    Returns:
        Up to ``n_results`` result dictionaries with the same schema as
        :func:`perform_similarity_search` (``similarity_score`` still describes
        semantic closeness, even when the list has been re-ordered by calories).
        An empty list when nothing matches or retrieval fails.

    Example:
        >>> [r["food_calories_per_serving"] for r in
        ...  perform_calorie_aware_search(collection, "deserts with lowest calroes")]
        [40, 90, 110, 120, 150]
    """
    constraints = parse_search_constraints(query, _known_cuisines(collection))
    has_constraint = bool(
        constraints["max_calories"]
        or constraints["sort_by_calories"]
        or constraints["cuisine_filter"]
    )
    if not has_constraint:
        return perform_similarity_search(collection, query, n_results)

    pool_size = max(candidate_pool, n_results)
    candidates = perform_filtered_similarity_search(
        collection,
        constraints["semantic_query"],
        cuisine_filter=constraints["cuisine_filter"],
        max_calories=constraints["max_calories"],
        n_results=pool_size,
    )

    # A cuisine *and* a calorie cap together can legitimately match nothing;
    # relaxing the numeric cap is friendlier than showing an empty screen.
    if not candidates and constraints["max_calories"] and constraints["cuisine_filter"]:
        candidates = perform_filtered_similarity_search(
            collection,
            constraints["semantic_query"],
            cuisine_filter=constraints["cuisine_filter"],
            n_results=pool_size,
        )

    if not candidates:
        return []

    if constraints["sort_by_calories"]:
        # Stable sort: equal-calorie dishes keep their semantic ordering.
        candidates.sort(key=lambda item: item.get("food_calories_per_serving") or 0)

    return candidates[:n_results]