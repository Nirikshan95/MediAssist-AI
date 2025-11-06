from langgraph.graph import StateGraph,START,END
from typing import Annotated
from utils.model_loader import load_model
from config.settings import GPT_OSS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class SymptomCollectorState:
    symptoms: list[str]
    messages: list[HumanMessage]
        
class SymptomCollectorSchema(BaseModel):
    question: Annotated[str, "The question to ask the user about their symptoms"]

parser = PydanticOutputParser(pydantic_object=SymptomCollectorSchema)


class SymptomCollector:
    def __init__(self):
        self.chat_model = load_model(model_id=GPT_OSS,temperature=0.2)
        self.prompt = open("prompts/symptoms_collector_prompt.txt","r").read()
    
    def symptoms_collect(self,state):
        print(f"symptoms :\n\n{state['symptoms']}\n\n messages:\n\n{state['messages']}")
        prompt_template = PromptTemplate(
            input_variables=["symptoms"],
            template=self.prompt,
            partial_variables={"format_instructions":parser.get_format_instructions()}
        )
        chain=prompt_template|self.chat_model|parser
        response=chain.invoke({"symptoms":state["symptoms"]})
        print("Symptom Collector Response:",response.question)
        #print("\n\n state messages:",state["messages"])
        for msg in state["messages"][::-1]:
            if isinstance(msg,HumanMessage):
                self.symptom=msg.content
                break
            else:
                self.symptom=None
                print("No HumanMessage found in messages.")
                
        return {"messages":AIMessage(content=response.question),"symptoms":[self.symptom] if self.symptom else []}
def Create_symptoms_collector_agent(state):
    print("we are in symptoms collector agent")
    graph=StateGraph(state)
    graph.add_node("SymptomCollector",SymptomCollector().symptoms_collect)
    graph.add_edge(START,"SymptomCollector")
    graph.add_edge("SymptomCollector",END)
    symptoms_collector_graph=graph.compile(name="SymptomCollectorAgent")
    return symptoms_collector_graph