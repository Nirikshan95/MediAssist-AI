from pyexpat.errors import messages
from langgraph.graph import StateGraph,END,add_messages
from typing import TypedDict,Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.model_loader import load_model
from config.settings import BIO_MED_GPT
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class SymptomCollectorState(TypedDict):
    symptoms: Annotated[list[str], "List of symptoms reported by the user"]
    messages: Annotated[list[str], add_messages]
        
class SymptomCollectorSchema(BaseModel):
    question: Annotated[str, "The question to ask the user about their symptoms"]

parser = PydanticOutputParser(pydantic_object=SymptomCollectorSchema)
class SymptomCollector:
    def __init__(self,):
        self.chat_model = load_model(model_id=BIO_MED_GPT)
        self.prompt = open("prompts/symptoms_collector_prompt.txt","r").read()
        
    
    def collect_symptoms(self,state: SymptomCollectorState) -> SymptomCollectorState:
        prompt_template = PromptTemplate(
            input_variables=["symptoms"],
            template=self.prompt,
            partial_variables={"format_instructions":parser.get_format_instructions()}
        )
        chain=prompt_template|self.chat_model|parser
        
        return state
    

def create_symptom_collector(llm,prompt):
    pass
    