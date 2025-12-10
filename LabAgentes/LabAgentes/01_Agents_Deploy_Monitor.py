# Databricks notebook source
# MAGIC %md
# MAGIC ![](./Images/image.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Laboratório de Agentes com GenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criação do Agente por Código

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] databricks-openai uv openai
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "catalogo_databricks"
schema = "agentes_ia"
# se necessário usar external location
# external_location="abfss://dados@baraldistorage.dfs.core.windows.net/"

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# se necessário usar external location
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog} MANAGED LOCATION '{external_location}'")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} MANAGED LOCATION '{external_location}'")

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC     convert_to_openai_messages,
# MAGIC )
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC system_prompt = "You are a helpful assistant that can run Python code."
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC # Below, we add the `system.ai.python_exec` UDF, which provides
# MAGIC # a python code interpreter tool to our agent
# MAGIC # You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools
# MAGIC
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# MAGIC # List to store vector search tool instances for unstructured retrieval.
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # To add vector search retriever tools,
# MAGIC # use VectorSearchRetrieverTool and create_tool_info,
# MAGIC # then append the result to TOOL_INFOS.
# MAGIC # Example:
# MAGIC # VECTOR_SEARCH_TOOLS.append(
# MAGIC #     VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # )
# MAGIC
# MAGIC tools.extend(VECTOR_SEARCH_TOOLS)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ):
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: AgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: AgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         "Convert from ChatCompletion dict to Responses output item dictionaries"
# MAGIC         for message in messages:
# MAGIC             message = message.model_dump()
# MAGIC             role = message["type"]
# MAGIC             if role == "ai":
# MAGIC                 if tool_calls := message.get("tool_calls"):
# MAGIC                     return [
# MAGIC                         self.create_function_call_item(
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                             call_id=tool_call["id"],
# MAGIC                             name=tool_call["name"],
# MAGIC                             arguments=json.dumps(tool_call["args"]),
# MAGIC                         )
# MAGIC                         for tool_call in tool_calls
# MAGIC                     ]
# MAGIC                 else:
# MAGIC                     return [
# MAGIC                         self.create_text_output_item(
# MAGIC                             text=message["content"],
# MAGIC                             id=message.get("id") or str(uuid4()),
# MAGIC                         )
# MAGIC                     ]
# MAGIC             elif role == "tool":
# MAGIC                 return [
# MAGIC                     self.create_function_call_output_item(
# MAGIC                         call_id=message["tool_call_id"],
# MAGIC                         output=message["content"],
# MAGIC                     )
# MAGIC                 ]
# MAGIC             elif role == "user":
# MAGIC                 return [message]
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
# MAGIC
# MAGIC         for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
# MAGIC             if event[0] == "updates":
# MAGIC                 for node_data in event[1].values():
# MAGIC                     for item in self._langchain_to_responses(node_data["messages"]):
# MAGIC                         yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
# MAGIC             # filter the streamed messages to just the generated text messages
# MAGIC             elif event[0] == "messages":
# MAGIC                 try:
# MAGIC                     chunk = event[1][0]
# MAGIC                     if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(delta=content, item_id=chunk.id),
# MAGIC                         )
# MAGIC                 except Exception as e:
# MAGIC                     print(e)
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphResponsesAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

from agent import AGENT

result = AGENT.predict({"input": [{"role": "user", "content": "Quanto é 6*7?"}]})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

from agent import AGENT

# Exibir o draw_mermaid
from IPython.display import Image, display

display(Image(AGENT.agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-langchain",
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Quanto é 6*7?"}]},
    env_manager="uv",
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
model_name = "agentepython"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "docs"},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Request e Feedback
# MAGIC 1) Use o seguinte request:
# MAGIC
# MAGIC {
# MAGIC   "input": [
# MAGIC     {
# MAGIC       "role": "user",
# MAGIC       "content": "oi, tudo bem?"
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC
# MAGIC ![Screenshot 2025-10-29 at 17.22.18.png](./Images/screen03.png "Screenshot 2025-10-29 at 17.22.18.png")
# MAGIC
# MAGIC 2) Em seguida, vá até o playground e escolha o seu agente para interagir:
# MAGIC
# MAGIC ![Screenshot 2025-10-29 at 17.14.15.png](./Images/screen01.png "Screenshot 2025-10-29 at 17.14.15.png")
# MAGIC
# MAGIC 3) Por último, verifique o app de feedback:
# MAGIC ![Screenshot 2025-10-29 at 17.19.22.png](./Images/screen02.png "Screenshot 2025-10-29 at 17.19.22.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Request em endpoint e monitoramento em tabela do sistema junto a guardrails com as respostas formatadas

# COMMAND ----------

from openai import OpenAI
import os
import getpass

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
#DATABRICKS_TOKEN = getpass.getpass("Databricks token: ")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-3250511655996160.0.azuredatabricks.net/serving-endpoints"
)

response = client.responses.create(
    model="agents_catalogo_databricks-agentes_ia-agentepython",
    input=[
        {
            "role": "user",
            "content": "Oi, tudo bem? O que é o Databricks?"
        }
    ]
)

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC 1) Verifique que a atividade no endpoint é monitorada:
# MAGIC ![Screenshot 2025-10-29 at 17.52.14.png](./Images/screen05.png "Screenshot 2025-10-29 at 17.52.14.png")
# MAGIC
# MAGIC 2) Vá até a aba Experiments e perceba que o tracing está lá e também é possível criar guidelines para LLM as a Judge. Vamos criar uma com o nome de no_customers e a guideline "A resposta não deve ter menções a concorrentes do mercado financeiro"
# MAGIC ![Screenshot 2025-10-29 at 17.48.00.png](./Images/screen04.png "Screenshot 2025-10-29 at 17.48.00.png")
# MAGIC
# MAGIC 3) Verifique o uso da avaliação (o job deve levar alguns minutos para executar):
# MAGIC ![Screenshot 2025-12-08 at 14.29.31.png](./Images/screenevaluation.png "Screenshot 2025-12-08 at 14.29.31.png")
# MAGIC
# MAGIC 3) Use a seguinte query no SQL Editor para verificar as requests. Caso queira, entre na aba Jobs & Pipelines ver o Job agendado que atualiza tais métricas:
# MAGIC
# MAGIC SELECT
# MAGIC   request,
# MAGIC   response,
# MAGIC   get_json_object(response, '$.output[2].content[0].text') AS output_text,
# MAGIC   request_time
# MAGIC FROM catalogo_databricks.agentes_ia.agentepython_payload
# MAGIC ORDER BY request_time DESC limit 1
# MAGIC
# MAGIC ![Screenshot 2025-10-30 at 13.04.32.png](./Images/screen06.png "Screenshot 2025-10-30 at 13.04.32.png")