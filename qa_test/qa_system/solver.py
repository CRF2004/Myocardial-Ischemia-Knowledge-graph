solver_prompt = """
    You are a medical reasoning module.

    Your task is to output a MINIMAL Reasoning Graph in STRICT JSON and answer the multiple-choice question.

    The Reasoning Graph is a structured representation of your key reasoning steps that can be verified or challenged by external evidence.

    IMPORTANT RULES:
    - Output JSON only. No extra text.
    - Each claim must be a simple, checkable medical statement.
    - Use concise, guideline-style medical expressions.

    INPUT QUESTION:
    {question}

    OPTIONS (letter -> text):
    {options}

    OUTPUT FORMAT (exact fields only):

    {
    "final_answer": "A|B|C|D",
    "task_type": "diagnosis|management|treatment|risk|mechanism|other", //Use only these types

    "entities": [
        {
        "id": "E1",
        "type": "Symptoms|DiagnosticTest|Disease|TreatmentIntervention|PatientCharacteristics", //Use only these types
        "text": "..."
        }
    ],

    "claims": [
        {
        "id": "C1",
        "premises": ["E1", "E2"],
        "relation": "INDICATES|SUPPORTS|RECOMMENDS|CONTRADICTS|REQUIRES|CAUSES|ASSOCIATED_WITH|...",
        "conclusion": "E3"
        }
    ],
    "option_mapping": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
    }
    }

    CONSTRAINTS:
    - Every premise and conclusion must reference an entity id.
    - At least one claim must connect findings/symptoms/tests to a diagnosis or decision.
    - At least one claim must distinguish or rule out a competing option.
    - Do NOT invent numeric thresholds unless explicitly stated in the question.

    Now output the Reasoning Graph JSON only.
"""