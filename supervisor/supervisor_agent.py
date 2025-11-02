from langgraph.graph import StateGraph,END,add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from typing import TypedDict,Annotated
import sqlite3

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list[str],add_messages]

# database connetion
conn=sqlite3.connect('supervisor_agent.db',check_same_thread=False)
memory=SqliteSaver(conn=conn)

def create_supervisor_agent(agents,llm,prompt):
    supervisor=create_supervisor(supervisor_name="MedAssist",agents=agents,model=llm,prompt=prompt)
    supervisor.compile(checkpointer=memory)
    return supervisor