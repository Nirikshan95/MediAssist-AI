from langgraph.graph import StateGraph,START,END
from typing import Annotated
from utils.model_loader import load_model
from config.settings import GPT_OSS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

        
class DiagnosisSchema(BaseModel):
    need_more_symptoms: Annotated[bool, "Whether more symptoms are needed to diagnose the issue"]
    diagnosis: Annotated[str, "The diagnosis of the user about their symptoms"]
    

parser = PydanticOutputParser(pydantic_object=DiagnosisSchema)


class Diagnosis:
    def __init__(self):
        self.chat_model = load_model(model_id=GPT_OSS,temperature=0.2)
        self.prompt = open("prompts/diagnosis_prompt.txt","r").read()
    
    def diagnosize(self,state):
        print(f"symptoms :\n\n{state['symptoms']}\n\n messages:\n\n{state['messages']}")
        prompt_template = PromptTemplate(
            input_variables=["messages","symptoms"],
            template=self.prompt,
            partial_variables={"format_instructions":parser.get_format_instructions()}
        )
        chain=prompt_template|self.chat_model|parser
        response=chain.invoke({"symptoms":state["symptoms"],"messages":state["messages"][-8:]})
        print("\n\n Diagnosis need more symptoms:\n",response.need_more_symptoms,"\n diagnosis:",response.diagnosis)
        return {
            "messages":[AIMessage(content=response.question)],
            "diagnosis":[response.diagnosis] if response.diagnosis else [None],
            "need_more_symptoms":[response.need_more_symptoms] if response.need_more_symptoms else [False]}
def Create_diagnosis_agent(state):
    print("we are in diagnosis agent")
    graph=StateGraph(state)
    graph.add_node("Diagnosis",Diagnosis().diagnosize)
    graph.add_edge(START,"Diagnosis")
    graph.add_edge("Diagnosis",END)
    diagnosis_graph=graph.compile(name="DiagnosisAgent")
    return diagnosis_graph