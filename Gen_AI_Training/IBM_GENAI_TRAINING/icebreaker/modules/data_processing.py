"""Prepare LinkedIn profile data for semantic search.

This module contains the processing steps between receiving a profile as a
Python dictionary and making that profile available through a LlamaIndex
vector index:

1. :func:`split_profile_data` serializes the dictionary as JSON, wraps it in a
   LlamaIndex :class:`~llama_index.core.Document`, and splits the document
   into nodes.
2. :func:`create_vector_database` embeds those nodes and builds a
   :class:`~llama_index.core.VectorStoreIndex`.
3. :func:`verify_embeddings` checks whether the index's vector store returns
   a non-``None`` value for each node ID recorded by the index.

The functions report failures through logging and return sentinel values
instead of propagating exceptions: an empty list for splitting, ``None`` for
index creation, and ``False`` for verification. Callers should check these
results before passing data to the next step.

Example:
    Process a profile after obtaining its data from an extraction step. The
    embedding model used during index creation must be configured for the
    current environment (for example, with the required watsonx credentials).

    >>> from modules.data_processing import (
    ...     split_profile_data,
    ...     create_vector_database,
    ...     verify_embeddings,
    ... )
    >>> profile = {
    ...     "first_name": "Ada",
    ...     "last_name": "Lovelace",
    ...     "headline": "Mathematician and computing pioneer",
    ...     "experiences": [{"title": "Analyst", "company": "Example Institute"}],
    ... }
    >>> nodes = split_profile_data(profile)
    >>> if nodes:
    ...     index = create_vector_database(nodes)
    ...     if index is not None:
    ...         embeddings_present = verify_embeddings(index)

The example intentionally does not assume how many nodes a profile produces:
that depends on the serialized content and the configured chunk size.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from modules.llm_interface import create_watsonx_embedding
import config

logger = logging.getLogger(__name__)

def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Serialize a profile dictionary and split its text into LlamaIndex nodes.

    The input is encoded with :func:`json.dumps` using its default settings.
    The resulting JSON text is placed in one LlamaIndex ``Document`` and
    divided by ``SentenceSplitter`` using ``config.CHUNK_SIZE``. The nodes
    retain the document text and metadata provided by LlamaIndex and can be
    passed to :func:`create_vector_database`.

    Args:
        profile_data: Profile fields and values represented as a dictionary.
            Values must be serializable by :func:`json.dumps`; nested
            dictionaries, lists, strings, numbers, booleans, and ``None`` are
            typical JSON-compatible values.

    Returns:
        A list of LlamaIndex nodes when serialization and splitting succeed.
        If either step raises an exception, the error is logged and an empty
        list is returned. An empty list can therefore indicate either a
        processing failure or that no nodes were produced.

    Example:
        >>> profile = {
        ...     "first_name": "Ada",
        ...     "headline": "Mathematician",
        ...     "skills": ["mathematics", "computing"],
        ... }
        >>> nodes = split_profile_data(profile)
        >>> isinstance(nodes, list)
        True

    Note:
        This function does not validate the profile schema or remove empty
        fields. It processes the dictionary as supplied. Values that are not
        JSON-serializable (such as an arbitrary Python object) cause the
        logged failure path and result in ``[]``.
    """
    try:
        # Convert the profile data to a JSON string
        profile_json = json.dumps(profile_data)

        # Create a Document object from the JSON string
        document = Document(text=profile_json)

        # Split the document into nodes using SentenceSplitter
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
        nodes = splitter.get_nodes_from_documents([document])
        
        logger.info(f"Created {len(nodes)} nodes from profile data")
        return nodes
    except Exception as e:
        logger.error(f"Error in split_profile_data: {e}")
        return []

def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Embed document nodes and build a LlamaIndex vector index.

    The embedding model is obtained from
    :func:`modules.llm_interface.create_watsonx_embedding`; LlamaIndex then
    uses it while constructing a ``VectorStoreIndex`` from the supplied
    nodes. Despite the function name, this returns a LlamaIndex index object;
    it does not accept a database path or configure persistence in this
    function.

    Args:
        nodes: Nodes to index, normally the list returned by
            :func:`split_profile_data`. This function does not validate the
            contents of the list before passing it to LlamaIndex.

    Returns:
        The constructed ``VectorStoreIndex`` if embedding and index
        construction succeed. If obtaining the embedding model or creating
        the index raises an exception, the error is logged and ``None`` is
        returned.

    Example:
        >>> nodes = split_profile_data({
        ...     "first_name": "Ada",
        ...     "headline": "Mathematician",
        ... })
        >>> index = create_vector_database(nodes)  # Requires configured watsonx access.
        >>> index is None or isinstance(index, VectorStoreIndex)
        True

    Note:
        Index creation may invoke the embedding provider and can therefore
        require network access, valid provider credentials, and a usable
        watsonx embedding configuration. The function logs failures rather
        than raising them to its caller.
    """
    try:
        # Get the embedding model
        embedding_model = create_watsonx_embedding()

        # Create a VectorStoreIndex from the nodes
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_model,
            show_progress=True
        )
        
        logger.info("Vector database created successfully")
        return index
    except Exception as e:
        logger.error(f"Error in create_vector_database: {e}")
        return None

def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Check that the index's vector store has an entry for every indexed node.

    The function reads node IDs from the index structure and calls
    ``vector_store.get(node_id)`` for each one. A ``None`` result is treated
    as a missing embedding; any non-``None`` result is treated as present.
    This is a presence check, not a validation of embedding dimensions,
    numeric values, semantic quality, or successful retrieval for a query.

    Args:
        index: LlamaIndex vector index whose stored node entries should be
            checked. It is expected to have the storage context and index
            structure used by the configured LlamaIndex implementation.

    Returns:
        ``True`` if every recorded node ID returns a non-``None`` value from
        the vector store. Returns ``False`` if at least one entry is missing
        or if verification raises an exception; failures are logged.

    Example:
        >>> index = create_vector_database(nodes)  # Requires configured watsonx access.
        >>> if index is not None:
        ...     all_entries_present = verify_embeddings(index)

    Implementation detail:
        This check uses ``index._storage_context`` and
        ``index.index_struct``, including a private storage-context
        attribute. Its behavior therefore depends on those LlamaIndex
        implementation details remaining available.
    """
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True
            else:
                logger.debug(f"Node ID {node_id} has a valid embedding.")
        
        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
        else:
            logger.info("All node embeddings are valid")
            return True
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False