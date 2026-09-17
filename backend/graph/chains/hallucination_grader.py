from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in the generated answer.
    """
    
    binary_score:bool = Field(
        description = "Answer is grounded in the facts, 'yes' or 'no'"
    )
    
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. 
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

"""
The hallucination grader is perhaps the most critical component for ensuring the reliability of our RAG system. 
It compares the generated response against the retrieved documents to verify that the information is factually grounded.
This prevents the system from generating plausible-sounding but factually incorrect responses. 
The grader uses a Boolean score to indicate whether the generation is supported by the provided facts.
When hallucinations are detected, our system can trigger regeneration or seek additional information, ensuring that users receive accurate and trustworthy responses.
"""