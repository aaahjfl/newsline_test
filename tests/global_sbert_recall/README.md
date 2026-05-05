# Global SBERT Recall Experiment

This folder archives the rejected experiment that built one persistent embedding
index for all news titles and used a topic embedding for first-stage candidate
recall.

It is not part of the production event-discovery pipeline. The production path
returns to NLLB topic alias expansion, SQL `LIKE` candidate recall, and SBERT
clustering only inside the recalled candidate set.
