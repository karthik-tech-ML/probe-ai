# Scenario Schema

## Overview
Each scenario in the library represents a single test case for the ProbeAI evaluation framework. Scenarios are the atomic unit of testing — every eval run pulls from this library.

## Schema Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier, format: `sc_XXX` |
| `question` | string | yes | The natural language question to ask the RAG system |
| `expected_answer` | string | yes | The correct/expected answer (used for scoring) |
| `category` | enum | yes | One of: `simple_factual`, `multi_field`, `aggregation`, `no_answer`, `adversarial`, `multi_hop` |
| `expected_source_ids` | list[int] | yes | TMDB movie IDs that should be retrieved. Empty list for `no_answer` scenarios. |
| `difficulty` | enum | yes | One of: `easy`, `medium`, `hard` |
| `hallucination_risk` | enum | yes | One of: `low`, `medium`, `high` |
| `notes` | string | no | Explanation of what this scenario tests |

## Categories

### simple_factual
Single-fact questions with one clear answer from one movie document.
Example: "Who directed Inception?"

### multi_field
Questions requiring information from multiple fields within one or more movie documents.
Example: "What Christopher Nolan movies are in the sci-fi genre?"

### aggregation
Questions requiring comparison or aggregation across multiple movies.
Example: "What is the highest-grossing action movie in the dataset?"

### no_answer
Questions about movies or facts NOT in the dataset. The system should acknowledge it doesn't have the information rather than hallucinate.
Example: "What is the plot of Oppenheimer?"

### adversarial
Questions designed to trigger hallucination or test grounding. May contain false premises or ask for information not present in the source data.
Example: "What Oscar did Inception win for Best Picture?"

### multi_hop
Questions requiring retrieval from multiple documents and reasoning across them.
Example: "Which actors appeared in both a Christopher Nolan film and a Quentin Tarantino film?"
