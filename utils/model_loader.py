from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


def load_model(model_id:str,temperature:float=0.3):
    endpoint=HuggingFaceEndpoint(model=model_id,temperature=temperature)
    model=ChatHuggingFace(llm=endpoint)
    return model