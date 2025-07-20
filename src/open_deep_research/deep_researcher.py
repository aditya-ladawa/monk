from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.configuration import (
    Configuration, 
)
from open_deep_research.state import (
    AgentState,
    AgentInputState,
    SupervisorState,
    ResearcherState,
    ClarifyWithUser,
    ResearchQuestion,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    lead_researcher_prompt
)
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_api_key_for_model,
    get_notes_from_tool_calls
)

from dotenv import load_dotenv
load_dotenv()

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(model_config)
    response = await model.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str()))])
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


async def write_research_brief(state: AgentState, config: RunnableConfig)-> Command[Literal["research_supervisor"]]:
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    response = await research_model.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    ))])
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=lead_researcher_prompt.format(
                        date=get_today_str(),
                        max_concurrent_research_units=configurable.max_concurrent_research_units
                    )),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    lead_researcher_tools = [ConductResearch, ResearchComplete]
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    # Exit Criteria
    # 1. We have exceeded our max guardrail research iterations
    # 2. No tool calls were made by the supervisor
    # 3. The most recent message contains a ResearchComplete tool call and there is only one tool call in the message
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    # Otherwise, conduct research and gather results.
    try:
        all_conduct_research_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"]
        conduct_research_calls = all_conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow_conduct_research_calls = all_conduct_research_calls[configurable.max_concurrent_research_units:]
        researcher_system_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
        coros = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    SystemMessage(content=researcher_system_prompt),
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config) 
            for tool_call in conduct_research_calls
        ]
        tool_results = await asyncio.gather(*coros)
        tool_messages = [ToolMessage(
                            content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for observation, tool_call in zip(tool_results, conduct_research_calls)]
        # Handle any tool calls made > max_concurrent_research_units
        for overflow_conduct_research_call in overflow_conduct_research_calls:
            tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_conduct_research_call["id"]
            ))
        raw_notes_concat = "\n".join(["\n".join(observation.get("raw_notes", [])) for observation in tool_results])
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": [raw_notes_concat]
            }
        )
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.research_model):
            print(f"Token limit exceeded while reflecting: {e}")
        else:
            print(f"Other error in reflection phase: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    research_model = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    # NOTE: Need to add fault tolerance here.
    response = await research_model.ainvoke(researcher_messages)
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    # Early Exit Criteria: No tool calls (or native web search calls)were made by the researcher
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        return Command(
            goto="compress_research",
        )
    # Otherwise, execute tools and gather results.
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    tool_calls = most_recent_message.tool_calls
    coros = [execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls]
    observations = await asyncio.gather(*coros)
    tool_outputs = [ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for observation, tool_call in zip(observations, tool_calls)]
    
    # Late Exit Criteria: We have exceeded our max guardrail tool call iterations or the most recent message contains a ResearchComplete tool call
    # These are late exit criteria because we need to add ToolMessages
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls):
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
            }
        )
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
        }
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesis_attempts = 0
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    researcher_messages = state.get("researcher_messages", [])
    # Update the system prompt to now focus on compression rather than research.
    researcher_messages[0] = SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    while synthesis_attempts < 3:
        try:
            response = await synthesizer_model.ainvoke(researcher_messages)
            return {
                "compressed_research": str(response.content),
                "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
            }
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                print(f"Token limit exceeded while synthesizing: {e}. Pruning the messages to try again.")
                continue         
            print(f"Error synthesizing research report: {e}")
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
    }


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []},}
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    findings = "\n".join(notes)
    max_retries = 3
    current_retry = 0
    while current_retry <= max_retries:
        final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            findings=findings,
            date=get_today_str()
        )
        try:
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)])
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                if current_retry == 0:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                print("Reducing the chars to", findings_token_limit)
                findings = findings[:findings_token_limit]
                current_retry += 1
            else:
                # If not a token limit exceeded error, then we just throw an error.
                return {
                    "final_report": f"Error generating final report: {e}",
                    **cleared_state
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [final_report],
        **cleared_state
    }

# deep_researcher_builder = StateGraph(AgentState, config_schema=Configuration)
# # deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
# deep_researcher_builder.add_node("write_research_brief", write_research_brief)
# deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
# deep_researcher_builder.add_node("final_report_generation", final_report_generation)
# # deep_researcher_builder.add_edge(START, "clarify_with_user")
# deep_researcher_builder.add_edge(START, "research_supervisor")
# deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
# deep_researcher_builder.add_edge("final_report_generation", END)

# open_deep_research_team = deep_researcher_builder.compile(name='open_deep_researcher')

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_tavily import TavilySearch
from langgraph_supervisor import create_supervisor

from open_deep_research.tools import *

from dotenv import load_dotenv


load_dotenv()

llm = init_chat_model(model = "gpt-4.1-mini", model_provider= "openai")

tavily_tool = TavilySearch(
    max_results=5,)


# === Skill Discovery Agent ===
skill_discovery_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt="""
You are the Skill Discovery Agent.

Your task:
- Extract 3 high-potential skills from the user context.
- For each skill, provide:
  - Market demand level (High/Medium/Low)
  - Learning curve (Easy/Moderate/Hard)
  - Competition level (High/Medium/Low)
  - Monetization potential and strategy

Output format:
Skill Name: XYZ
- Market Demand: High
- Learning Curve: Moderate
- Competition: Low
- Monetization Strategy: ...


ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.
""",
    name="skill_discovery_agent"
)

# === Skill Analyzer Agent ===
skill_analyzer_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt="""
You are the Skill Analyzer Agent.

Your task:
- Take each skill from discovery agent and validate its feasibility.
- Design a lean 30–60 day learning plan per skill.
- Use only recent, highly effective resources (videos, tutorials, docs).
- Provide learning sequence, milestones, and resource links.

Output format:
Skill: XYZ
- Week 1: ...
- Week 2: ...
- Resources: [list of clickable links]

ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.


""",
    name="skill_analyzer_agent"
)

# === Skill Supervisor ===
skill_supervisor = create_supervisor(
    model=llm,
    agents=[skill_discovery_agent, skill_analyzer_agent],
prompt="""
You are the Skill Supervisor.

Use the agents as follows:

Agent Name: skill_discovery_agent  
When to use: When the user provides context or goals to identify 3 high-potential skills with their market demand, learning curve, competition, and monetization potential.

Agent Name: skill_analyzer_agent  
When to use: After skills have been discovered, to validate each skill and design a 30–60 day learning plan with effective resources and milestones.

ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.

""",
).compile(name="skill_supervisor")

# === Teacher Agent ===
teacher_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt="""
You are the Teacher Agent.

You answer general user questons for the given topic. You have web search tool for internet search

Constraints:
- Verify resource quality before recommending.
- Ask user for clarification if the skill is vague.
- No padding, encouragement, or repetition.
- Output as clean markdown, one skill per section.

ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.

""",
    name="teacher_agent"
)

# === Structured File Writer Agent ===
structured_file_writer_agent = create_react_agent(
    llm,
    tools=[write_md_document, edit_md_document, read_md_document, write_json_document, read_json_document],
    prompt=(
        "You write, edit, and read structured documents (Markdown and JSON).\n"
        "Synthesize planning and analysis outputs into clean, valid files.\n"
        "No follow-up questions."
        "ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR."
    ),
    name="structured_file_writer_agent"
)

# === Structured Outliner Agent ===
structured_outliner_agent = create_react_agent(
    llm,
    tools=[create_md_outline, read_md_document],
    prompt="""
        "You generate structural markdown outlines for complex documents like roadmaps.\n"
        "No follow-up questions.",
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    """,
    name="structured_outliner_agent",
)

# === Chart Generator Agent ===
chart_generator_agent = create_react_agent(
    llm,
    tools=[read_md_document, python_repl_tool],
    prompt=(
        "You generate charts and visualizations via Python code.\n"
        "Return markdown embeddable image references or base64 images.\n"
        "No follow-up questions."
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    ),
    name="chart_generator_agent"
)

# === General Document Writing Supervisor ===
general_doc_writer_team = create_supervisor(
    model=llm,
    agents=[
        structured_file_writer_agent,
        structured_outliner_agent,
        chart_generator_agent
    ],
    prompt=(
        "You manage:\n"
        "- structured_outliner_agent: document structure outlines\n"
        "- chart_generator_agent: visualizations\n"
        "- structured_file_writer_agent: .md/.json writing and editing\n\n"
        "Workflow:\n"
        "1. Generate outline if needed\n"
        "2. Generate visuals if needed\n"
        "3. Compile final document\n\n"
        "Rules:\n"
        "- Delegate all work\n"
        "- Never generate content yourself\n"
        "- Deliver clean, complete documents"
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    ),
).compile(name='general_doc_writer_team')

# === Skill Roadmap Planner Agent ===
skill_roadmap_planner_agent = create_react_agent(
    model=llm,
    tools=[parse_progress_data],
    prompt="""
You are the Skill Roadmap Planner Agent.

Tasks:
- Parse user progress data files.
- Draft 3–6 month actionable learning roadmap aligned with career goals.
- Output clean comprehensive markdown roadmap with timeline and skills.

Constraints:
- Use only provided data.
- No explanations, only roadmap markdown output.
ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.

""",
    name="skill_roadmap_planner_agent"
)

# === Monetization Planner Agent ===
monetization_planner_agent = create_react_agent(
    model=llm,
    tools=[parse_progress_data, tavily_tool],
    prompt="""
You are the Monetization Planner Agent.

Tasks:
- Parse roadmap and skills data.
- Use Tavily to find freelancing, product, niche job opportunities.
- Map 2-3 monetization pathways with:
  - Execution timeline
  - Platforms
  - Estimated revenue range

Format:
### Monetization Strategy 1
- Type: Freelancing
- Skills Used: ...
- Platform: ...
- Revenue: $X–$Y/month
- Timeline: ...

No theory, no fluff, only execution plans.
Provide comprehensive monitization plan.
ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.

""",
    name="monetization_planner_agent"
)

# === Roadmap Writing Team Supervisor ===
roadmap_writing_team_supervisor = create_supervisor(
    model=llm,
    agents=[
        skill_roadmap_planner_agent,
        monetization_planner_agent,
    ],
prompt="""
You supervise the Roadmap Writing Team.

Use the agents as follows:

Agent Name: skill_roadmap_planner_agent  
When to use: When user progress data is available and a clear, actionable 3–6 month learning roadmap aligned with career goals is needed.

Agent Name: monetization_planner_agent  
When to use: After the learning roadmap is created, to generate monetization strategies based on the roadmap and skills, including timelines, platforms, and revenue estimates.
ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.
""",
).compile(name="roadmap_writing_team_supervisor")

# === Researcher Team ===
research_agent = create_react_agent(
    llm,
    tools=[write_md_document],
    prompt=(
        "You are a researcher. Given a query, gather detailed info and write raw markdown to 'findings.md'. "
        "Respond ONLY with the file path."
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    ),
    name="research_agent"
)

note_taker_agent = create_react_agent(
    llm,
    tools=[read_md_document, create_md_outline],
    prompt=(
        "You are a note taker. Read 'findings.md', create key points outline, save as 'notes.md'. "
        "Respond ONLY with the file path."
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    ),
    name="note_taker_agent"
)

report_writer_agent = create_react_agent(
    llm,
    tools=[read_md_document, write_md_document],
    prompt=(
        "You are a report writer. Read 'notes.md', write final report markdown to 'final_report.md'. "
        "Respond ONLY with the file path."
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    ),
    name="report_writer_agent"
)

deep_research_supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, note_taker_agent, report_writer_agent],
    prompt=(
        "You are Deep Research Supervisor.\n"
        "Protocol:\n"
        "1. Assign research_agent to collect raw data.\n"
        "2. Assign note_taker_agent to summarize findings.\n"
        "3. Repeat if notes insufficient.\n"
        "4. Assign report_writer_agent to produce final report.\n"
        "Respond only with agent commands and file paths.\n"
        "No direct research or reasoning."
        'ONCE DONE. ALWAYS REPORT BACK TO SUPERVISOR.'
    )
).compile(name="deep_research_supervisor")

# === Final System Supervisor ===
graph = create_supervisor(
    model=llm,
    agents=[
        skill_supervisor,
        deep_research_supervisor,
        teacher_agent,
        structured_outliner_agent,
        roadmap_writing_team_supervisor,
        monetization_planner_agent,
    ],
    prompt="""
You are the System Supervisor responsible for orchestrating the full skill-to-income roadmap workflow.

you can go to / use any agent or team you see fit for the given task. Below are the details

skill_supervisor team:

skill_discovery_agent: When the user provides context or goals to identify 3 high-potential skills.

skill_analyzer_agent: After skills are discovered, to validate and design 30–60 day learning plans.

Teacher Agent

teacher_agent: For answering general user questions about skills with verified, quality learning paths and resources.

General Document Writing Supervisor Team

structured_outliner_agent: When you need structured markdown outlines for complex documents.

chart_generator_agent: When visual charts or data visualizations are required.

structured_file_writer_agent: To write or edit Markdown/JSON documents and compile final files.

roadmap_writing_team_supervisor team:

skill_roadmap_planner_agent: When user progress data is available and a clear 3–6 month learning roadmap is needed.

monetization_planner_agent: After roadmap creation, to build monetization strategies with timelines, platforms, and revenue estimates.

deep_research_supervisor team:

research_agent: To gather raw, detailed information from queries into markdown files.

note_taker_agent: To create outlines of key points from raw research.

report_writer_agent: To write polished reports from notes.



Dont't mention agent names or team names. Make user feel as if he is talking to one agent. In background you will handle any delgations and receive repsonse. You should only output the final consie answer.
Since you will be used as an voice agent, be sure to answer in under 3 concise sentences. You should basically summarize what output we got or what the agent or team did.
""",
).compile(name="system_supervisor")