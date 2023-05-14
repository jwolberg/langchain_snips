#!/usr/bin/env python



import os
import re
import sys
from _ast import If
sys.path.insert(0, 'libs')
import logging
import requests
from requests.exceptions import HTTPError
import random
import pandas as pd
from flask import Flask, render_template, request, session, redirect, make_response, Response
import datetime
from datetime import date
from datetime import datetime as dt
from pytz import timezone
import openai
from typing import List

import langchain
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory, ChatMessageHistory
from langchain.schema import AIMessage, SystemMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from sentence_transformers import SentenceTransformer

from urllib.parse import urlparse, urldefrag, parse_qs, urlencode


from api_keys import  OPENAI_API_KEY

# from google.appengine.api import users

#====================================
from google.cloud import datastore
from google.appengine.api import users
from google.appengine.api import memcache, wrap_wsgi_app

#====================================

sys.path.append(os.path.join(os.pardir,os.getcwd()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # needed for sentense_transformer
os.environ["LANGCHAIN_TRACING"] = "true"  # trace execution of program

openai.api_key = OPENAI_API_KEY 
openai.organization = "org-xxxxxxxxxxxxxxx"


version = "0.5.00.01-openai"
PROJECT_NAME = 'Trading Volatility'

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

utc = timezone('UTC')
eastern = timezone('US/Eastern')


def ndb_wsgi_middleware(wsgi_app):
    def middleware(environ, start_response):
        with client.context():
            return wsgi_app(environ, start_response)

    return middleware


app = Flask(__name__)
app.wsgi_app = wrap_wsgi_app(app.wsgi_app)   #  creates WSGI middleware that sets the variables required to enable  API calls to google.appengine.api for memcache
app.wsgi_app = ndb_wsgi_middleware(app.wsgi_app)  # Wrap the app in middleware for ndb


app.secret_key = OPENAI_API_KEY

       


@app.route('/agent2', methods=['GET', 'POST'])
def agent2():
    siteTitle = 'Agent'
    history = ChatMessageHistory()
    conversation_history = session.get('conversation_history')
    human_content = ""
    ai_content = ""    

    if request.method == 'POST':
        
        question = request.form['question-input']
        clear_history = request.form['clear-history']        

        if clear_history == "1":
                                session.pop('conversation_history', None)
                                output = "Goodbye!"

        # Check for and load memory
        print("line 99 conversation_history: %s\n" % conversation_history)
        if conversation_history is not None:
            # Extract previous conversation into homan_content and ai_content, to add to history
            human_content = [d['data']['content'] for d in conversation_history if d['type'] == 'human']
            ai_content = [d['data']['content'] for d in conversation_history if d['type'] == 'ai']
            print(f"line 104 input_str={human_content}")
            print(f"line 105 output_str={ai_content}")
            length = len(human_content)
            n=0
            while n < length:
                history.add_user_message(human_content[n])
                history.add_ai_message(ai_content[n])
                n=n+1
            print("line 133 Imported history from conversation_history %s\n" % history)
           
            
        # Set Language model parameters
        llm= OpenAI(  
            model_name= "text-davinci-003", #model_name='gpt-3.5-turbo',
            temperature=0.0,
            max_tokens= 200, 
            openai_api_key=OPENAI_API_KEY)
        

    
        ### Define your prompts 
               


        ### Define memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        #import memory from previous conversations
        memory.save_context({"input": str(human_content)}, {"ouput": str(ai_content)})
        print("line 153 IMPORTED MEMORY FROM CONVERSATION HISTORY:  %s\n" % memory)
        
        ### Read Only Memory (If you want a "Summarize this conversation" tool 
        readonlymemory = ReadOnlySharedMemory(memory=memory)


        # Define Tools

        # create an instance of the custom langchain tools, as necessary
        
        # List of available tools
        tools = [
            
        ]

        # Add the current question to history
        history.add_user_message(question)

        
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        agent = PlanAndExecute(planner=planner, executer=executor, verbose=True)
       
        print ("line 176 history%s\n" % (history))
        print ("line 177 question%s\n" % (question))
        output = "<pre style='white-space: pre-wrap'>" + agent.run(question) + "</pre>"
    
        # Add the current Answer to history
        history.add_ai_message(output)
        conversation_history_dicts = messages_to_dict(history.messages)
        session['conversation_history'] = conversation_history_dicts
        print ("line 183 session %s\n" % (session))

    else: # a GET request 
        print ("line 186 empty session%s\n" % (session))
        session.clear()
        session.pop('conversation_history', None)
        output = ""



    return render_template('agent2.html', 
                            output=output,
                            session = session,
                            siteTitle = siteTitle, 
                            projectname = PROJECT_NAME
                            )


