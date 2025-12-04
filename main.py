import os
import requests
import wikipedia
import time
from typing import Annotated, Sequence, TypedDict, Literal
from datetime import datetime
import sys

from langchain_litellm import ChatLiteLLM  # 使用新包
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.tree import Tree
from rich import print as rprint


if os.name == 'nt':  # For windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

console = Console()
if not os.getenv("NEBIUS_API_KEY"):
    os.environ["NEBIUS_API_KEY"] = "Your_API_key"


geolocator = Nominatim(user_agent="advanced-agent-v1")


# 工具 1: 天气查询
class SearchInput(BaseModel):
    location: str = Field(description="The name of the city, e.g. San Francisco, Berlin")
    date: str = Field(description="The date for the weather forecast in yyyy-mm-dd format")


@tool("get_weather_forecast", args_schema=SearchInput, return_direct=False)
def get_weather_forecast(location: str, date: str):
    try:
        loc = geolocator.geocode(location)
        if loc:
            url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={loc.latitude}&longitude={loc.longitude}&"
                   f"hourly=temperature_2m&start_date={date}&end_date={date}")
            response = requests.get(url)
            data = response.json()
            if "hourly" in data:
                temps = data["hourly"]["temperature_2m"]
                times = data["hourly"]["time"]
                indices = [12, 18] if len(times) > 18 else [0]
                summary = {times[i]: temps[i] for i in indices}
                return f"Weather data for {location} on {date}: {summary} (Degrees Celsius)"
            return "Error: No hourly data found."
        return "Error: Location not found."
    except Exception as e:
        return f"API Error: {str(e)}"


@tool("search_wikipedia")
def search_wikipedia(query: str):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "No wikipedia page found."


tools = [get_weather_forecast, search_wikipedia]
tools_by_name = {tool.name: tool for tool in tools}


MODEL_ID = "nebius/Qwen/Qwen3-235B-A22B-Instruct-2507"

llm = ChatLiteLLM(model=MODEL_ID, temperature=0.1)
model = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



def call_tool(state: AgentState):
    outputs = []
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls'):
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tree = Tree(f"🛠️ [bold cyan]Executing Tool:[/bold cyan] {tool_name}")
            tree.add(f"Args: {tool_args}")
            console.print(tree)

            tool_result = tools_by_name[tool_name].invoke(tool_call["args"])

            time.sleep(0.5)

            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
            console.print(f"[dim]   -> Result: {str(tool_result)[:100]}...[/dim]\n")

    return {"messages": outputs}


def call_model(state: AgentState, config: RunnableConfig):
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")

    system_prompt = SystemMessage(
        content=f"You are a helpful assistant. Today's date is {current_date}. When checking weather, always use this date unless the user specifies otherwise.")


    messages_to_send = [system_prompt] + list(state["messages"])

    # 显示思考中的动画
    with console.status("[bold green]🤖 AI is thinking...", spinner="dots"):
        response = model.invoke(messages_to_send, config)

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"


workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)

workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "llm")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    rprint(Panel.fit(
        "[bold yellow]⚡ NEBIUS AI AGENT V2.0 ⚡[/bold yellow]\n"
        "[dim]Powered by LangGraph & Qwen/Llama[/dim]",
        border_style="blue"
    ))

    config = {"configurable": {"thread_id": "user-session-001"}}

    while True:
        try:
            user_input = console.input("\n[bold green]You > [/bold green]")
            if user_input.lower() in ["exit", "quit", "q"]:
                rprint("[bold red]Goodbye![/bold red]")
                break

            if not user_input.strip():
                continue

            inputs = {"messages": [("user", user_input)]}

            rprint("[line]")

            final_response = None
            for event in graph.stream(inputs, config=config, stream_mode="values"):
                message = event["messages"][-1]
                if isinstance(message, AIMessage) and not message.tool_calls:
                    final_response = message.content

            if final_response:
                rprint(
                    Panel(Markdown(final_response), title="[bold blue]AI Response[/bold blue]", border_style="green"))

        except Exception as e:
            console.print_exception()