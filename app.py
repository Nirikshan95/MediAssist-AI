import streamlit as st
from supervisor.supervisor_agent import agent_supervisor,save_workflow_visualization
from langchain_core.messages import HumanMessage

st.title("MediAssist-AI")
st.write("Welcome to MediAssist-AI, your personal medical assistant powered by AI.")
save_workflow_visualization()
input_text = st.chat_input("Describe your symptoms or medical concerns:")

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    
    supervisor = agent_supervisor()
    response = supervisor.invoke(
        {"messages": [HumanMessage(content=input_text)], "symptoms": []}, 
        config={"configurable": {"thread_id": "mahesh95"}}
    )
    
    with st.chat_message("assistant"):
        st.markdown(response["messages"][-1].content)