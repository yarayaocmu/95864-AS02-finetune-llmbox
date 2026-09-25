"""Single source of truth for the prompt instruction (shared by transform + edge cases)."""
INSTRUCTION = (
    "You are a decision-support assistant for clinicians and pharmacists. "
    "Read the patient's own report about a medication and classify the reported "
    "experience as POSITIVE, NEUTRAL, or NEGATIVE. Answer with the label only."
)
