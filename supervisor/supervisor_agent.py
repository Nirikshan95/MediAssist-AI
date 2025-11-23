from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from agents.symptom_collector_agent import Create_symptoms_collector_agent
from agents.diagnosis_agent import Create_diagnosis_agent
from typing import TypedDict, Annotated
from config.settings import GPT_OSS
from utils.model_loader import load_model
import sqlite3
import os
import operator
# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Changed from list[str] to list
    symptoms: Annotated[list, operator.add]  # Added symptoms field
    diagnosis: str
    need_more_symptoms: bool

# database connection
conn = sqlite3.connect('supervisor_agent.db', check_same_thread=False)
memory = SqliteSaver(conn=conn)

def agent_supervisor():
    workflow = create_supervisor(
        supervisor_name="MediAssist Supervisor",
        agents=[
            Create_symptoms_collector_agent(AgentState),
            Create_diagnosis_agent(AgentState)
        ],
        model=load_model(model_id=GPT_OSS),
        prompt=open("prompts/supervisor_prompt.txt", "r").read(),
        parallel_tool_calls=True
    )
    supervisor = workflow.compile(checkpointer=memory)
    return supervisor


def save_workflow_visualization():
    # 1. Get the compiled graph (app)
    app = agent_supervisor()
    
    print("Generating graph visualization...")
    try:
        # 2. Generate the PNG image using Mermaid API
        # This requires an internet connection to reach mermaid.ink
        graph_image = app.get_graph().draw_mermaid_png()
        
        # 3. Save to a file
        output_file = "workflow_graph.png"
        with open(output_file, "wb") as f:
            f.write(graph_image)
            
        print(f"✅ Successfully saved workflow graph to {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating graph: {e}")
        # Fallback: Print the Mermaid syntax
        print("\nMermaid Syntax (copy to https://mermaid.live):")
        print(app.get_graph().draw_mermaid())

if __name__ == "__main__":
    save_workflow_visualization()