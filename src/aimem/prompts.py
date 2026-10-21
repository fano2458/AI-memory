# PAPER_*, FULL_CONTEXT_* and BENCHMARK_* from mem0ai/mem0 and mem0ai/memory-benchmarks (Apache-2.0); placeholders changed to {x}.

PAPER_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}

    Question: {question}

    Answer:
    """

BENCHMARK_ANSWER_PROMPT = """You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
"""

PAPER_JUDGE_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

STRICT_JUDGE_PROMPT = """Label a generated answer as CORRECT or WRONG against a gold answer. The question asks about facts from past conversations between two people.

Mark CORRECT only if the generated answer states the same facts as the gold answer:
1. Every item in the gold answer must be present. If the gold lists several items, missing any one of them is WRONG. A reason after a semicolon in the gold ("Beach; he loves swimming") is an explanation, not a required item.
2. Dates must match at the level of detail the gold gives. If the gold gives a day, the day must match; if it gives a month or a year, the month or year must match.
3. If the gold is a period anchored to a date ("the week before 14 March 2023", "the friday before 2 April 2023"), a date inside that period, or the same relative expression, is CORRECT. If the gold is relative with no anchor ("5 years ago"), the same expression, or a year that fits it for conversations held in 2022-2024, is CORRECT.
4. Numbers, counts and durations must match exactly.
5. Names and entities must be the same ones, at the same level of detail. A related but different person, place or thing is WRONG, and so is a more general answer ("novels" for "mystery novels", "a car brand" for "Toyota"). A more specific answer that unambiguously implies the gold is CORRECT ("Lyon" for "France").
6. For questions that ask for a judgement or likelihood, the generated answer must reach the same conclusion as the gold ("Yes" matches "Likely yes").

Also mark WRONG if the generated answer:
- says the information is not available, or declines to answer;
- offers several alternatives without committing to the gold one;
- adds a claim that contradicts the gold.

Extra details that do not contradict the gold are fine. Wording and format may differ.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON with keys "reasoning" (one sentence) and "label" (CORRECT or WRONG).
"""

ADVERSARIAL_JUDGE_PROMPT = """The question below cannot be answered from the conversations: it asks about something that was never said, usually by attributing an event to the wrong person. The correct response is to say the information is not available.

Label the generated answer:
- ABSTAIN: it does not accept the false premise. It says the information is not mentioned or not available, declines to answer, denies the premise ("Sam has no pets"), or says the fact belongs to someone else ("Alex adopted it, not Jordan").
- TRAP: it accepts the premise and gives the trap answer, or the same fact in other words, for the person the question names, even if it also hedges.
- OTHER: it accepts the premise but gives a different answer.

Question: {question}
Trap answer: {adversarial_answer}
Generated answer: {generated_answer}

Return JSON with keys "reasoning" (one sentence) and "label" (ABSTAIN, TRAP or OTHER).
"""

FULL_CONTEXT_SYSTEM = 'You are a helpful assistant that can answer questions based on the provided context.If the question involves timing, use the conversation date for reference.Provide the shortest possible answer.Use words directly from the conversation when possible.Avoid using subjects in your answer.'

FULL_CONTEXT_PROMPT = """
# Question:
{question}

# Context:
{context}

# Short answer:
"""

SUFFICIENCY_JUDGE_PROMPT = """Decide whether the notes below are enough to answer a question. Each note starts with the date it was written.

Label YES if a careful reader of these notes alone could give the gold answer, or a clearly equivalent one. If the gold lists several items, at least one of them must be supported. A relative date counts when the note's date lets you work it out ("(12 March 2023) Sam started guitar last week" supports "the week before 12 March 2023").

Label NO if the needed fact is missing, too vague ("Sam enjoys music" does not support "guitar"), attributed to a different person, or dated so that the gold date cannot be worked out.

Only the gold answer must be supported. Details that appear in the question but not in the gold answer do not need to be in the notes.

Do not use outside knowledge about the people involved.

Notes:
{notes}

Question: {question}
Gold answer: {gold_answer}

Return JSON with keys "reasoning" (one sentence) and "label" (YES or NO).
"""

# STALE_* from icedreamc/STALE (MIT), STALE/Evaluation.

STALE_QUESTION_SYSTEM = 'You are a helpful assistant. Review the following conversation history with the user, then accurately answer the question.'

STALE_TASK_SYSTEM = "You are a helpful assistant. Review the following conversation history with the user, then respond to the user's latest query directly."

STALE_QUESTION_USER = "[Conversation History]\n{history}\n\n[Question]\n{query}"

STALE_TASK_USER = "[Conversation History]\n{history}\n\n[Latest Query]\n{query}"

STALE_JUDGE_SYSTEM = """
You are a strict and impartial expert evaluator for an AI Assistant Benchmark focusing on Long-Term Memory and Implicit Reasoning.

You will be provided with the Ground Truth Context of a user's state. 
Then, you will evaluate THREE distinct responses generated by a Target AI Model to THREE different probing questions (Dimension 1, 2, and 3).

[Ground Truth Context]:
- M_old: The user's outdated state.
- M_new: The user's updated state.
- Hidden Logic: How M_new invalidates M_old.

You must grade each response strictly based on whether it correctly deduces and respects the hidden updated state.

### Scoring Rubrics (Boolean: true for Pass, false for Fail)

* **Dimension 1 (Explicit Probing):** Score `true` ONLY IF Response 1 awares that the information in M_old is possibly NO LONGER VALID. Score `false` if it assumes M_old is still true or claims ignorance.

* **Dimension 2 (Adversarial Robustness):**
  Question 2 contains a FALSE PREMISE based on M_old. Score `true` ONLY IF Response 2 detects the trap, or acknowledges the new state in M_new. Score `false` if it blindly agrees with the false premise.

* **Dimension 3 (Implicit Task):**
  Score `true` ONLY IF the final action, plan, or recommendation in Response 3 is safely adheres to the NEW state established by M_new.  
  Score `false` if the action complies with the outdated M_old constraints, OR if the response is too generic, vague.

Evaluate strictly based on the underlying logic, not surface-level word matching.

### Output Format (JSON)
{
  "dim1_eval": {
    "reasoning": "Brief step-by-step reasoning against the Dim 1 rubric.",
    "pass": true/false
  },
  "dim2_eval": {
    "reasoning": "Brief step-by-step reasoning against the Dim 2 rubric.",
    "pass": true/false
  },
  "dim3_eval": {
    "reasoning": "Brief step-by-step reasoning against the Dim 3 rubric.",
    "pass": true/false
  }
}
"""

STALE_JUDGE_USER = """
[Ground Truth Context]
- M_old: "{old_info}"
- M_new: "{M_new}"
- Hidden Logic: {explanation}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {dim1_q}
Target Model Response 1: {dim1_a}

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {dim2_q}
Target Model Response 2: {dim2_a}

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {dim3_q}
Target Model Response 3: {dim3_a}
"""

# ZEP_* from mem0ai/mem0 evaluation (Apache-2.0); placeholders changed to {x}.

ZEP_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {memories}

    Question: {question}
    Answer:
    """

ZEP_CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}

"""

# Ours: ladder rung 1, added to each system's own answer prompt.

NEWEST_INSTRUCTION = "Some memories may be outdated, because facts about a person change over time. When memories about the same thing disagree, the most recent one is the current state and the older ones are history. Before answering, check whether the question assumes something that a more recent memory contradicts; if it does, say so and answer from the current state."

# Ours: ladder rung 2, a decision step before answering.

DECIDE_PROMPT = """Below are memories about people, numbered and dated, oldest first. Facts about people change: a later memory can make an earlier one outdated, either directly (a new value for the same thing) or by implication (a later change that means an earlier statement no longer holds).

List every memory that a later memory makes outdated, together with the number of that later memory. Do not list a memory only because a later one adds something alongside it or is about the same topic. Use the question only to see which memories matter; the question itself may assume something outdated.

Question: {question}

Memories:
{memories}

Return JSON: {{"outdated": [{{"id": <number>, "replaced_by": <number>}}]}}, or {{"outdated": []}} if nothing is outdated."""

# Ours: ladder rung 3, statuses decided at write time (scripts/replay.py).

KEY_PROMPT = """Give each memory below a short key naming what it says about someone, written as "subject | aspect". An aspect is one property of the subject that has a value at any given time, for example "Sam | job", "Sam | pets" or "Sam | daily schedule"; avoid broad themes such as "interests" or "routine". Memories about the same aspect of the same subject must get the same key, so reuse an existing key whenever one fits and make a new key only for a new aspect.

If a memory only records what was discussed, asked, offered or recommended in a conversation (for example "User was recommended ..." or "User asked about ..."), and states no fact about the person, give it the key null.

Existing keys:
{keys}

Memories:
{memories}

Return JSON: {{"keys": {{"0": "<key or null>", "1": "<key or null>"}}}} with an entry for every memory number."""

REPLACE_PROMPT = """Each new memory below is shown with older memories that have the same key. For each older memory, decide whether the new memory replaces it, meaning the older memory is no longer true, or whether both are still true because the new one repeats it, adds to it, or describes another part of it. Only a new value of the same property replaces an older one (a new job replaces the old job). A different trip, purchase, gift, event, person or item of the same kind does not replace an earlier one; both stay true.

{pairs}

Return JSON: {{"replaced": [{{"old": <O number>, "new": <N number>}}]}}, or {{"replaced": []}} if nothing is replaced."""
