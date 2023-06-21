 import os
import pinecone
import environ
env = environ.Env()
environ.Env.read_env()
from loguru import logger
from langchain.llms import OpenAI
from langchain import LLMChain,OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor,Tool,ZeroShotAgent
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores.faiss import FAISS
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
openai_api_key = env('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')




class Generate_QA_Agent:
    """Curriculum Agent"""

    def __init__(self):
        logger.debug("QA Agent init")
        self.llm = OpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        self.tools = None
        self.agent = None
        self.agent_chain=None
        self.llm_chain=None
        self.load_tools()
        self.load_agent()

    def load_tools(self):
        """
        Loading Tools for the Agent
        """

# if you have Pdf documents only
        # folder_path = "./documents"
        # file_list = os.listdir(folder_path)

        # documents = []
        # for file_name in file_list:
        #     file_path = os.path.join(folder_path, file_name)
        #     # print("Reading file:", file_path) 
        #     loader = PyPDFLoader(file_path)
        #     loaded_documents = loader.load()
        #     documents.extend(loaded_documents)
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(documents)
        # vector_store = FAISS.from_documents(texts, self.embeddings)

# if you have already created vectorstore of documents
        pinecone.init(api_key = pinecone_key, environment = pinecone_env)
        vector_store = Pinecone.from_existing_index("light-chapter-index", self.embeddings)
        retriever = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=vector_store.as_retriever()
        )
        
        self.tools = [
        Tool(
        name="QA Generator",
        func=retriever.run,
        description="""
        useful for when you need to generate subjective questions with answers as best 
        you can based on the context and memory available.
        """,
        ),
        Tool(name="MCQ Generator",
            func=retriever.run,
            description="""
            useful for when you need to generate mcq questions with 4 options and asnwers as best 
            you can based on the context and memory available.
            """,
        ),  
        ]

    def load_agent(self):
        """
        Loading Agent with Tools
        """
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        prefix = """You are a helpful assistant that generate and display questions with answers related to the vectorestore.
        If the input is not related to the provided document, say "Please ask question related to light only"
        You should not reach the itreation limit, after 3 iteration display, "Please ask question related to document only".
        Provide correct response, before finishing the chain.
        You should not generate MCQs and subjective questions outside the scope of the document,"say I dont have any information.


        Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.
        ### Instruction:

            Answer the following questions as best you can. You have access to the following tools:

            MCQ Generator: useful for when you need to generate multiple choice questions(MCQs) with four options and answers as best 
            you can based on the context and memory available.
            QA Generator: useful for when you need to generate questions with answers as best you can based on
            context and memory available.

            Strictly use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [MCQ Generator, QA Generator]
            Action Input: the input to the action, should be a question or instruction
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat 3 times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            For examples:
            Question: generate 2 MCQs with answers on incident light
            Thought: I need to generate 2 MCQs with options and answers on incident light.
            Action: MCQ Generator
            Action Input: Incident light
            Observation:  Incident light is the light that is incident on a surface, such as a mirror.
            Action: MCQ Generator
            Action Input: generate 2 MCQs with answers on incident light.
            Observation:  Incident light is the light that is incident on a surface, such as a mirror.
            Thought: I now have the information to generate 2 MCQs with four options and one answer on incident light.
                
            Final Answer:
            1. What is incident light or ray?
            A. Light that is reflected off a surface
            B. Light that is emitted from a source
            C. Light that is absorbed by a surface
            D. Light that is incident on a surface
            Answer: D. Light that is incident on a surface

            2. What is the angle of incidence?
            A. The angle between the incident light and the normal to the surface
            B. The angle between the reflected light and the normal to the surface
            C. The angle between the emitted light and the normal to the surface
            D. The angle between the absorbed light and the normal to the surface
            Answer: A. The angle between the incident light and the normal to the surface

            Question: Generate 1 subjective questions with answers on braille system
            Thought: I need to generate 1 subjective questions with answers on braille system
            Action: QA Generator
            Action Input: Braille system
            Observation:  Braille system is a system developed by Louis Braille in 1821 for visually challenged persons. It has 63 dot patterns or characters, each representing a letter, a combination of letters, a common word or a grammatical sign. The patterns are embossed on Braille sheets and help visually challenged persons to recognize words by touching. Visually impaired people learn the Braille system by beginning with letters, then special characters and letter combinations. Braille texts can be produced by hand or by machine, such as typewriter-like devices and printing machines.
            Thought:I now have the information to generate 3 subjective questions with answers on braille system.
            Action: QA Generator
            Action Input: Braille system
            Observation:  Braille system is a system developed by Louis Braille in 1821 for visually challenged persons. It has 63 dot patterns or characters, each representing a letter, a combination of letters, a common word or a grammatical sign. The patterns are embossed on Braille sheets and help visually challenged persons to recognize words by touching. Visually impaired people learn the Braille system by beginning with letters, then special characters and letter combinations. Braille texts can be produced by hand or by machine, such as typewriter-like devices and printing machines.
            Thought:I now have the subjective questions with answers on braille system
            Final Answer:
            1. What is the Braille system and how does it work?
            Answer: The Braille system is a system developed by Louis Braille in 1821 for visually challenged persons. It has 63 dot patterns or characters, each representing a letter, a combination of letters, a common word or a grammatical sign. The patterns are embossed on Braille sheets and help visually challenged persons to recognize words by touching. Visually impaired people learn the Braille system by beginning with 
            letters, then special characters and letter combinations.

            Question: Generate 3 subjective questions on respiration
            Observation: respiration is not related to the context provided.
            Thought: Please ask question related to light only.
            Observation: Invalid Format: Missing 'Action:' after 'Thought:'
            Thought:Please ask question related to light only.
            Final Answer: Please ask a question related to light only.

            ### Input:
            {{question}}
            ### Response:
            Question: {{question}}
            Thought: {{gen 'thought' stop='\\n'}}
            Action: {{select 'tool_name' options=valid_tools}}
            Action Input: {{gen 'actInput' stop='\\n'}}
            Observation:{{search actInput}}
            Thought: {{gen 'thought2' stop='\\n'}}
            Final Answer: {{gen 'final' stop='\\n'}}""" 
        
        suffix="""Begin!"
        {chat_history}
        Questions:{input}
        {agent_scratchpad}
        """
    
        prompt =ZeroShotAgent.create_prompt(
                self.tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input","chat_history","agent_scratchpad"]
                
            )
            
        self.llm_chain= LLMChain(
            llm=OpenAI(temperature=0,openai_api_key=openai_api_key,model_name="gpt-3.5-turbo"),
                prompt=prompt,
                
        )
    
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain,tools=self.tools,verbose=True,max_iterations=4,max_execution_time=10,)
        self.agent_chain=AgentExecutor.from_agent_and_tools(
            agent=self.agent,tools=self.tools,verbose=True, memory=memory,handle_parsing_errors=True
    )
        

