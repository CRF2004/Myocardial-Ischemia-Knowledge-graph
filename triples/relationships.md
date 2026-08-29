##### 1. **Core Medical Relationships**

- `TREATS` - Treatment relationship (Medical Interventions→Disease, Drugs→Disease, Medical Interventions→Symptoms, Drugs→Symptoms)
- `DIAGNOSES` - Diagnostic relationship (Diagnostic Methods→Disease, Diagnostic Methods→Pathological Markers)
- `INDICATES` - Indicative relationship (Symptoms→Disease, Biomarkers→Disease, Pathological Markers→Disease, Diagnostic Methods→Disease)
- `CAUSES` - Causal relationship (Disease→Symptoms, Pathological Markers→Symptoms, Disease→Pathological Markers)

##### 2. General Association Relationships

- `ASSOCIATED_WITH` - Association relationship (General association between any entities)
- `MEASURES` - Measurement relationship (Diagnostic Methods→Biomarkers, Diagnostic Methods→Pathological Markers)
- `AFFECTS` - Influence relationship (Drugs→Biomarkers, Medical Interventions→Biomarkers)

##### 3. Temporal/Process Relationships

- `PRECEDES` - Precedes (Temporal relationship between any entities)
- `PREVENTS` - Prevention (Drugs→Disease, Medical Interventions→Disease)

##### 4. Structural Relationships

- `PART_OF` - Part-of relationship (Symptoms are part of a disease, Pathological Markers are part of a disease process)

5. Exclusion / Contradiction Relationships

- `CONTRADICTS` - Exclusion or contradiction relationship
(Findings, diagnostic results, or clinical evidence that argue against a disease, diagnosis, or pathological state)