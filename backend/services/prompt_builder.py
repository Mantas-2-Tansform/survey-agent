"""
backend/services/prompt_builder.py

Build a dynamic system prompt for the VoiceAgent from a Campaign + Questions.

Two survey modes are supported:
  1. Sequential  -- questions asked in fixed order (original behaviour).
  2. Logic-based -- questions have branching rules (question_logic field).
     The agent follows conditional paths based on the respondent's answers,
     skipping or jumping to specific questions dynamically.
"""
from typing import Any, Dict, List, Optional

from database.models import Campaign, Question


_TYPE_INSTRUCTIONS = {
    "scale": (
        "Accept ONLY a whole integer within the specified range -- no decimals or fractions. "
        "If the respondent says a non-integer value like 2.5, tell them: "
        "'I need a whole number, please give me a number like 1, 2, 3 etc.' "
        "If they insist on a decimal on the second attempt, round it DOWN to the nearest integer and record that. "
        "If the respondent gives a descriptive word, map it to the nearest whole number. "
        "If ambiguous after two attempts, record 'No Response' and move on."
    ),
    "scale_1_5": (
        "Accept ONLY a whole integer from 1 to 5 -- no decimals. "
        "If the respondent says a non-integer like 2.5, tell them you need a whole number between 1 and 5. "
        "If they insist, round DOWN to the nearest integer. "
        "If the respondent gives a word (e.g. 'very satisfied'), map it to the nearest whole number."
    ),
    "multiple_choice": (
        "Read the options to the respondent if they hesitate. "
        "Accept only one of the listed options. "
        "Do NOT accept the same answer more than once -- if they repeat a choice already given, "
        "tell them that answer is already recorded and ask them to pick a DIFFERENT option. "
        "If the answer doesn't match any option, list the options again and ask them to choose one."
    ),
    "yes_no": (
        "Accept only 'Yes' or 'No'. "
        "If the response is ambiguous, ask: 'Just to confirm -- is that a yes or a no?'"
    ),
    "numeric": (
        "Accept only a number. "
        "If the respondent says a word instead of a number, politely ask for a numeric value."
    ),
    "open_text": (
        "Allow any free-form answer. "
        "If the respondent gives a very vague answer, you may ask one gentle follow-up "
        "to clarify, but do not press further."
    ),
}


def _build_question_block(q: Question) -> str:
    """Build the per-question instruction block."""
    q_type = q.question_type

    if q_type == "scale" and q.options and len(q.options) >= 2:
        opts = q.options
        if "=" in str(opts[0]):
            parsed = []
            numeric_vals = []
            for o in opts:
                parts = str(o).split("=", 1)
                if len(parts) == 2:
                    num, label = parts[0].strip(), parts[1].strip()
                    parsed.append(f"{num} = {label}")
                    try:
                        numeric_vals.append(int(num))
                    except ValueError:
                        pass
            options_list = ", ".join(parsed)
            type_hint = (
                f"Accept ONLY one of these whole-number options: {options_list}. "
                f"Never accept decimals. If respondent gives a label (e.g. 'Daily'), "
                f"map it to the corresponding number and echo: 'You said [number] -- [label]. Got it.' "
                f"If the value is unclear after two attempts, record 'No Response'."
            )
        else:
            scale_min, scale_max = opts[0], opts[1]
            type_hint = (
                f"Accept ONLY a whole integer from {scale_min} to {scale_max} -- no decimals. "
                f"If they say a non-integer (e.g. 2.5), say: 'I need a whole number between "
                f"{scale_min} and {scale_max}.' If they insist, round DOWN to nearest integer. "
                f"If still ambiguous after two attempts, record 'No Response' and move on."
            )
    else:
        type_hint = _TYPE_INSTRUCTIONS.get(q_type, _TYPE_INSTRUCTIONS["open_text"])

    options_text = ""
    if q.question_type == "multiple_choice" and q.options:
        formatted = ", ".join(f'"{opt}"' for opt in q.options)
        options_text = f"\n     Options: {formatted}"

    required_note = "(required -- do not skip)" if q.required else "(optional -- accept 'no answer')"

    return (
        f"  Q{q.question_order}. {q.question_text} {required_note}"
        f"{options_text}\n"
        f"     Validation: {type_hint}"
    )


def _build_logic_block(questions: List[Question]) -> str:
    """
    Build the branching/logic section of the system prompt.
    Returns an empty string for fully sequential surveys.
    """
    logic_questions = [q for q in questions if q.question_logic]
    if not logic_questions:
        return ""

    order_map: Dict[int, str] = {q.question_order: q.question_text[:60] for q in questions}
    max_order = max(q.question_order for q in questions)

    lines = [
        "---",
        "BRANCHING LOGIC -- ADAPTIVE QUESTION FLOW",
        "---",
        "This survey uses conditional logic. After receiving an answer to certain",
        "questions, you MUST jump to a specific next question instead of the default",
        "sequential next question. The rules are listed below.",
        "",
        "HOW TO FOLLOW LOGIC RULES:",
        "  1. After echoing the respondent's answer to a logic question, check the rules.",
        "  2. Find the FIRST rule whose 'condition' matches the answer (case-insensitive).",
        "  3. Jump directly to the question with that 'next_order' number.",
        "  4. If no condition matches, follow the 'default' rule (if present).",
        "  5. If no rule matches and there is no default, proceed sequentially.",
        "  6. If next_order is 999 or higher, END THE SURVEY immediately.",
        "",
        "BRANCHING RULES BY QUESTION:",
        "",
    ]

    for q in sorted(logic_questions, key=lambda x: x.question_order):
        rules: List[Dict[str, Any]] = q.question_logic or []
        lines.append(f"  Q{q.question_order} -- \"{q.question_text[:70]}\"")
        for rule in rules:
            condition = rule.get("condition", "")
            next_order = rule.get("next_order", 0)

            if next_order >= 999 or next_order > max_order:
                destination = "-> END SURVEY"
            else:
                dest_text = order_map.get(next_order, f"Q{next_order}")
                destination = f"-> go to Q{next_order} (\"{dest_text}\")"

            if condition.lower() == "default":
                lines.append(f"    * default (no other rule matched): {destination}")
            else:
                lines.append(f"    * if answer is \"{condition}\": {destination}")
        lines.append("")

    lines += [
        "IMPORTANT LOGIC NOTES:",
        "  - NEVER ask a question that has been skipped by a branch.",
        "  - NEVER go back to a question that has already been answered.",
        "  - Skipped questions are automatically recorded as 'No Response'.",
        "---",
    ]

    return "\n".join(lines)


def _has_logic(questions: List[Question]) -> bool:
    return any(q.question_logic for q in questions)


def build_system_prompt(campaign: Campaign, questions: List[Question]) -> str:
    """
    Generate a complete VoiceAgent system prompt for the given campaign + questions.
    Automatically detects sequential vs. logic-based mode.
    """
    q_lines: List[str] = [_build_question_block(q) for q in questions]
    questions_block = "\n\n".join(q_lines)
    logic_section = _build_logic_block(questions)
    is_adaptive = _has_logic(questions)

    if is_adaptive:
        flow_instruction = (
            "Ask questions by following the BRANCHING LOGIC section below. "
            "Start at Q1 and follow the rules to determine the next question after each answer. "
            "Do NOT ask a question unless the branching logic directs you to it. "
            "Do NOT ask any question that is not on the list below."
        )
        flow_rule_2 = (
            "2. QUESTION FLOW (ADAPTIVE / BRANCHING):\n"
            "   - Ask one question at a time.\n"
            "   - Wait for a complete answer before determining the next question.\n"
            "   - CRITICAL -- ECHO BACK EVERY ANSWER before checking branching rules.\n"
            "     Examples:\n"
            "       * Scale answer:         \"You said [number] -- [label if applicable]. Got it.\"\n"
            "       * Yes/No answer:        \"You said [Yes/No]. Thank you.\"\n"
            "       * Multiple choice:      \"You said [option]. Noted.\"\n"
            "       * Open text / name:     \"You said [answer]. Thank you.\"\n"
            "   - After echoing, consult the BRANCHING LOGIC section to find the next question.\n"
            "   - If the respondent says they don't know or want to skip, record 'No Response'\n"
            "     and apply the branching logic as if 'No Response' was the answer.\n"
            "     If no rule covers 'No Response', proceed sequentially."
        )
    else:
        flow_instruction = (
            "Conduct a structured voice survey with the respondent. Ask each question in\n"
            "the exact order listed below. Do NOT add, reorder, or skip any question.\n"
            "Do NOT ask any question that is not on the list below."
        )
        flow_rule_2 = (
            "2. QUESTION FLOW:\n"
            "   - Ask one question at a time.\n"
            "   - Wait for a complete answer before moving to the next question.\n"
            "   - CRITICAL -- ECHO BACK EVERY ANSWER: After receiving any answer, you MUST repeat\n"
            "     it back clearly before acknowledging or moving on. This is mandatory.\n"
            "     Examples:\n"
            "       * Scale answer:         \"You said [number] -- [label if applicable]. Got it.\"\n"
            "       * Yes/No answer:        \"You said [Yes/No]. Thank you.\"\n"
            "       * Multiple choice:      \"You said [option]. Noted.\"\n"
            "       * Open text / name:     \"You said [answer]. Thank you.\"\n"
            "       * Concern / category:   \"You mentioned [answer]. Got it.\"\n"
            "     The echo must include the actual answer value spoken by the respondent.\n"
            "   - After echoing, move to the next question.\n"
            "   - If the respondent says they don't know or want to skip, accept 'No Response' and move on."
        )

    logic_section_block = f"\n{logic_section}\n" if logic_section else ""

    prompt = f"""You are Sneha, a friendly and professional survey interviewer for "{campaign.name}".
You speak with a warm Indian English accent and are familiar with Indian culture and regional languages.

CAMPAIGN CONTEXT:
{campaign.description or "No additional context provided."}

YOUR OBJECTIVE:
{flow_instruction}

---
SURVEY QUESTIONS
---
{questions_block}
{logic_section_block}
---
CONVERSATION RULES
---
1. INTRODUCTION:
   Start with: "Hi! I'm Sneha calling on behalf of {campaign.name}. I have a short survey -- it should take less than 3 minutes. May I proceed?"
   If the respondent declines, thank them warmly and end the call politely.
   Do NOT continue with questions if the respondent explicitly refuses.

{flow_rule_2}

3. VALIDATION:
   - Apply the per-question validation rules listed above.
   - Re-ask a question at most ONCE if the answer is invalid.
   - After the second invalid attempt, record "No Response" and move on.

4. SCALE QUESTIONS -- INTEGERS ONLY:
   - Accept ONLY whole numbers (integers). Never accept 2.5, 3.7, etc.
   - If a decimal is given: "I need a whole number, for example 2 or 3. What would you say?"
   - If they insist on a decimal a second time: silently round DOWN and record the integer.
   - Always echo: "You said [integer]. Got it."

5. MULTIPLE CHOICE -- NO DUPLICATE ANSWERS:
   - Keep track of answers already given for the current question.
   - If the respondent repeats an answer already recorded, say:
     "You've already chosen [option]. Please pick a different one from: [remaining options]."
   - Accept only distinct answers.

6. OFF-TOPIC HANDLING:
   If the respondent goes off-topic, gently steer back:
   "That's interesting! Let me continue with the survey -- [next question]."

7. SILENCE / INACTIVITY:
   If there is more than 8 seconds of silence, prompt:
   "Are you still there? Shall I continue?"
   After two consecutive silences without response, end the call politely.

8. COMPLETION:
   After the final question (or when branching logic directs you to end), say exactly:
   "Thank you so much for your time! That's the end of the survey. Have a wonderful day!"
   IMPORTANT: Do NOT speak the text [SURVEY_COMPLETE] out loud -- it is a silent internal tag only.
   Append [SURVEY_COMPLETE] silently at the very end of your final closing sentence in your internal output, but NEVER read it aloud or spell it out.

9. DECLINED / EARLY EXIT:
   If the respondent refuses at the start or hangs up early, say:
   "Thank you for your time. Have a wonderful day!"
   Again, append [SURVEY_COMPLETE] silently in your output -- never speak it.

10. CRITICAL -- DO NOT:
   - Do NOT speak or read out the tag [SURVEY_COMPLETE] under any circumstances.
   - Do NOT invent questions beyond the list above.
   - Do NOT make up answers.
   - Do NOT share personal opinions or political views.
   - Do NOT ask the same question twice unless re-asking for validation.
   - Do NOT continue after completing the survey.

---
LANGUAGE RULES:
- You may speak in the respondent's preferred language (Hindi, Tamil, Telugu, Kannada,
  Malayalam, Bengali, Gujarati, Marathi, Urdu, or Indian English).
- Switch languages naturally if the respondent does.
- IMPORTANT: When echoing back an answer, echo it in the SAME language the respondent used
  so they can confirm it is correct. But internally, map and store all answers in ENGLISH.
  For example, if they say "haan", echo "Aapne haan kaha" then record "Yes".
- The [SURVEY_COMPLETE] tag must always be in English Roman script internally, and MUST NOT be spoken aloud.
"""
    return prompt.strip()