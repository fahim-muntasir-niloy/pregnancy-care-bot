import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import tool
from langgraph.prebuilt import  ToolNode, tools_condition, create_react_agent
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict, Annotated, Optional, Sequence
from system_prompt import system_prompt


from langchain_core.messages import (AnyMessage,
                                     AIMessage,
                                     SystemMessage,
                                     HumanMessage, 
                                     ToolMessage)

from tool import retrieve_relevant_info, search_web, llm




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        # MessagesPlaceholder(variable_name="messages"),
        ("human", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# === Create LLM Agent ===
tools = [
    retrieve_relevant_info,
    search_web  
         ]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, 
                               tools=tools,
                            #    return_intermediate_steps=True,
                               handle_parsing_errors=True,
                            #    verbose=True
                               )


# === Create State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    
    
def agent_node(state: AgentState) -> AgentState:
    result = agent_executor.invoke({"messages": state["messages"]})
    # If result is a string, wrap it as AIMessage
    if isinstance(result, str):
        ai_message = AIMessage(content=result)
    elif isinstance(result, dict) and "output" in result:
        ai_message = AIMessage(content=result["output"])
    else:
        ai_message = result  # if result is already a Message
        
    return {
        "messages": state["messages"] + [ai_message],
    }

    
tool_node = ToolNode(tools=tools)


# ==== graph ====
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent",
                                    tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

graph.name = "Pregnancy_Doc_Agent"

# print(agent_executor.invoke({"messages": [HumanMessage(content="৫ মাস চলছে। পেটের ভেতর বাচ্চা বেশী নড়ছে না মনে হচ্ছে.")]}))
