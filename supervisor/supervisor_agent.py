from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from agents.symptom_collector_agent import Create_symptoms_collector_agent
from typing import TypedDict, Annotated
from config.settings import GPT_OSS
from utils.model_loader import load_model
import sqlite3
import os

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Changed from list[str] to list
    symptoms: Annotated[list, add_messages]  # Added symptoms field

# database connection
conn = sqlite3.connect('supervisor_agent.db', check_same_thread=False)
memory = SqliteSaver(conn=conn)

def agent_supervisor():
    workflow = create_supervisor(
        supervisor_name="MediAssist Supervisor",
        agents=[Create_symptoms_collector_agent(AgentState)],
        model=load_model(model_id=GPT_OSS),
        prompt=open("prompts/supervisor_prompt.txt", "r").read(),
    )
    supervisor = workflow.compile(checkpointer=memory)
    return supervisor