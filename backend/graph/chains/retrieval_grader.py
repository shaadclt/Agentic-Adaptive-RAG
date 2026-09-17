from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    
    binary_score:str = Field(
        description = "Documents are relevant to the question, 'yes' or 'no'"
    )
    
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


"""
The retrieval grader acts as a quality control mechanism that evaluates whether retrieved documents are actually relevant to the user's question. 
This component is crucial because vector similarity alone doesn't guarantee relevance — documents might be semantically similar but contextually inappropriate.
The GradeDocuments model ensures we get a clean binary decision from the LLM. The system prompt instructs the grader to look for both explicit keywords and semantic meaning, providing a comprehensive relevance assessment. This grading step prevents irrelevant documents from contaminating our generation process and triggers web search when local documents are insufficient.
"""