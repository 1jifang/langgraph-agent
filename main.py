import os
import requests
import wikipedia
import time
import sqlite3
from typing import Annotated, Sequence, TypedDict, Literal
from datetime import datetime
import sys

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.tree import Tree
from rich import print as rprint
from rich.prompt import Confirm

if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

console = Console()
if not os.getenv("NEBIUS_API_KEY"):
    os.environ["NEBIUS_API_KEY"] = "Your_API_key"

geolocator = Nominatim(user_agent="advanced-agent-v1")


# --- 工具定义 (已修复) ---

class SearchInput(BaseModel):
    location: str = Field(description="The name of the city, e.g. San Francisco, Berlin")
    date: str = Field(description="The date for the weather forecast in yyyy-mm-dd format")

@tool("get_weather_forecast", args_schema=SearchInput, return_direct=False)
def get_weather_forecast(location: str, date: str):
    """
    Get the weather forecast for a specific location and date.
    Returns temperature data in Celsius.
    """
    # ^^^ 必须加上这一段 Docstring，否则 LangChain 会报错 ^^^
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

            try:
                tool_result = tools_by_name[tool_name].invoke(tool_call["args"])
            except Exception as e:
                tool_result = f"Tool execution failed: {e}"

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
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = SystemMessage(
        content=f"You are a helpful assistant. Today's date is {current_date}. When checking weather, always use this date unless the user specifies otherwise.")

    messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    messages_to_send = [system_prompt] + messages

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

db_path = "agent_state.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

if __name__ == "__main__":
    rprint(Panel.fit(
        "[bold yellow]⚡ NEBIUS AI AGENT V2.1 (HITL + Persistence) ⚡[/bold yellow]\n"
        "[dim]Powered by LangGraph & SQLite[/dim]",
        border_style="blue"
    ))

    thread_id = "user-session-persistent-001"
    config = {"configurable": {"thread_id": thread_id}}

    rprint(f"[dim]Session ID: {thread_id} (History is saved to agent_state.db)[/dim]")

    while True:
        try:
            snapshot = graph.get_state(config)
            if snapshot.next and "tools" in snapshot.next:
                last_msg = snapshot.values["messages"][-1]
                tool_calls = last_msg.tool_calls

                rprint("\n[bold red]⚠️  Approval Required[/bold red]")
                for tc in tool_calls:
                    rprint(f"AI wants to run: [bold cyan]{tc['name']}[/bold cyan] with args: {tc['args']}")

                is_approved = Confirm.ask("Do you approve this execution?")

                if is_approved:
                    rprint("[green]Tools Approved. Resuming...[/green]")
                    inputs = None
                else:
                    rprint("[red]Tools Rejected.[/red]")
                    tool_rejections = []
                    for tc in tool_calls:
                        tool_rejections.append(
                            ToolMessage(
                                tool_call_id=tc["id"],
                                content=f"User denied permission to execute tool {tc['name']}."
                            )
                        )
                    graph.update_state(config, {"messages": tool_rejections}, as_node="tools")
                    inputs = None

            else:
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
            time.sleep(1)
