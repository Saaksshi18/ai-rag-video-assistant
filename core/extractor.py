#Actionableitems , decision , questions 

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 



def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

# Shared guidance appended to every extraction prompt. The transcripts often
# come from YouTube videos, not just meetings, so the model has to be told to
# skip the creator's boilerplate and to answer in plain text the UI can parse.
_COMMON_RULES = (
    "\n\nRules:\n"
    "- Ignore promotional or channel-housekeeping lines such as asking viewers "
    "to like, subscribe, follow, comment, share, join a community, visit a "
    "sponsor, or 'stay tuned until the end'. They are never action items, "
    "decisions or open questions.\n"
    "- Use plain text only: no markdown, no ** bold, no backticks, no headings "
    "or introductory sentence. Start directly with item 1."
)


def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Only include concrete tasks that follow from the content itself (for a "
        "tutorial, the practical steps the audience is told to do; for a meeting, "
        "work people committed to). If nobody is named as responsible, write "
        "'Not specified' for Owner.\n\n"
        "Format as a numbered list, with each item written as: "
        "Task: ... Owner: ... Deadline: ... "
        "If none found say 'No action items found.'"
        + _COMMON_RULES
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
        + _COMMON_RULES
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
        + _COMMON_RULES
    )
    return chain.invoke(transcript)