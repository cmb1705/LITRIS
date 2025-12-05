---
description: Search the literature index
allowed-tools: Bash(python:*), Read
argument-hint: <query>
---

Search the literature index using semantic search.

Query: $ARGUMENTS

Execute the query_index.py script with the provided query.
Format results as markdown showing:
1. Ranked list of matching papers (title, authors, year, score)
2. Matching excerpts from each paper
3. Suggested related queries

If filters are mentioned in the query (e.g., "from 2020", "about methods"), apply appropriate filters.
