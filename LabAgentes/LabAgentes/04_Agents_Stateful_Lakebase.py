# Databricks notebook source
# MAGIC %md
# MAGIC ![](./Images/image.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Mosaic AI Agent Framework + Lakebase

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies

# COMMAND ----------

# DBTITLE 1,or
# MAGIC %pip install -U -qqqq databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] \
# MAGIC   langgraph-checkpoint-postgres==2.0.21 psycopg[binary,pool] bs4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## First time setup only: Set up checkpointer with your Lakebase instance

# COMMAND ----------

import os
import uuid
from databricks.sdk import WorkspaceClient
from psycopg_pool import ConnectionPool
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

DB_NAME = "databricks_postgres"
SSL_MODE = "require"
DB_HOST = "instance-6fbb9a7e-4a68-47e5-9b92-19b181e35f40.database.azuredatabricks.net"
DB_PORT = 5432
WORKSPACE_HOST = "https://adb-3250511655996160.0.azuredatabricks.net/"

# Seu usuário Databricks (identidade OAuth)
DB_USER = "daniel.baraldi@databricks.com"

# Pegue o token OAuth (ex.: via variável de ambiente)
OAUTH_TOKEN = """eyJraWQiOiJjMGQ2YzQ2MTA4NWVmY2E1YTgzYTMxNzI2ZDQ2ZmMzN2QxNmMwYzY4NWQwNDRhMTJhNTUxNjhhOGM3MzZkM2U2IiwidHlwIjoiYXQrand0IiwiYWxnIjoiUlMyNTYifQ.eyJjbGllbnRfaWQiOiJkYXRhYnJpY2tzLXNlc3Npb24iLCJzY29wZSI6ImlhbS5jdXJyZW50LXVzZXI6cmVhZCBpYW0uZ3JvdXBzOnJlYWQgaWFtLnNlcnZpY2UtcHJpbmNpcGFsczpyZWFkIGlhbS51c2VyczpyZWFkIiwiaWRtIjoiRUFBPSIsImlzcyI6Imh0dHBzOi8vYWRiLTMyNTA1MTE2NTU5OTYxNjAuMC5henVyZWRhdGFicmlja3MubmV0L29pZGMiLCJhdWQiOiIzMjUwNTExNjU1OTk2MTYwIiwic3ViIjoiZGFuaWVsLmJhcmFsZGlAZGF0YWJyaWNrcy5jb20iLCJpYXQiOjE3NjUyODU5NTksImV4cCI6MTc2NTI4OTU1OSwianRpIjoiYWQxZjQ1MzYtNjM0My00ODRmLWJiNDgtNjBkZjE2NmJhOGEzIn0.A8M1KbkATpETZUIauifYR-AneclokAllj-SWyE7LqbjAGGfd-lE_eA5AOV_TAQV4Vdt_N8YNs-ZCX2pj_I8SY_ruedMc-vU0hWsYi6urQfkptEOWROgbpk25DfTYveE0LszUOg2Vbky5LGGyvW4KBcEKFaAsPDDmBxc44sAyqywItsDSXARUZiKlrxP02rilix0I2F7Ggyfp_Jcbrta2-GwYJTq_aFikLWXJzvpNzhOgJdpWe6nerVbVU8ANDrKCQHT3m-mpU_LbA8ogRCTvzrk-ZDvHWo8pJFkbtTilVetpFEFTcS7vDpfyqqLB5vE_ADJmyyCl1zRx_LcKcOL9tw"""

# Autentica no workspace com o token OAuth do usuário
w = WorkspaceClient(host=WORKSPACE_HOST, token=OAUTH_TOKEN)

def db_password_provider() -> str:
    # Usa o token OAuth como "senha" do PostgreSQL
    # Se você tiver um fluxo que renova o token, prefira:
    # return w.config.oauth_token().access_token
    return OAUTH_TOKEN

class CustomConnection(psycopg.Connection):
    @classmethod
    def connect(cls, conninfo="", **kwargs):
        kwargs["password"] = db_password_provider()
        return super().connect(conninfo, **kwargs)

pool = ConnectionPool(
    conninfo=f"dbname={DB_NAME} user={DB_USER} host={DB_HOST} port={DB_PORT} sslmode={SSL_MODE}",
    connection_class=CustomConnection,
    min_size=1,
    max_size=10,
    open=True,
)

# Use the pool to initialize your checkpoint tables
with pool.connection() as conn:
    conn.autocommit = True  # disable transaction wrapping
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    conn.autocommit = False  # restore default if you want transactions later

    with conn.cursor() as cur:
        cur.execute("select 1")
        print("✅ Pool connected and checkpoint tables are ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the agent in code
# MAGIC
# MAGIC ## Write agent code to file agent.py
# MAGIC Define the agent code in a single cell below. This lets you write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC ## Wrap the LangGraph agent using the ResponsesAgent interface
# MAGIC For compatibility with Databricks AI features, the `LangGraphResponsesAgent` class implements the `ResponsesAgent` interface to wrap the LangGraph agent.
# MAGIC
# MAGIC Databricks recommends using `ResponsesAgent` as it simplifies authoring multi-turn conversational agents using an open source standard. See MLflow's [ResponsesAgent documentation](https://www.mlflow.org/docs/latest/llms/responses-agent-intro/).

# COMMAND ----------

# MAGIC %%writefile agentlakebase.py
# MAGIC import json
# MAGIC import logging
# MAGIC import os
# MAGIC import time
# MAGIC import uuid
# MAGIC from threading import Lock
# MAGIC from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, List
# MAGIC from contextlib import contextmanager
# MAGIC from urllib.parse import urlparse, parse_qs, unquote, quote
# MAGIC from io import BytesIO
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks, UCFunctionToolkit
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langgraph.checkpoint.postgres import PostgresSaver
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent
# MAGIC
# MAGIC import psycopg
# MAGIC from psycopg_pool import ConnectionPool
# MAGIC from psycopg.rows import dict_row
# MAGIC
# MAGIC # Cache core
# MAGIC from langchain_core.caches import BaseCache
# MAGIC from langchain_core.outputs import Generation
# MAGIC from langchain_core.globals import set_llm_cache
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
# MAGIC
# MAGIC ############################################
# MAGIC # LLM endpoint e system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to answer questions."
# MAGIC
# MAGIC ############################################
# MAGIC # LAKEBASE CONFIG
# MAGIC ############################################
# MAGIC LAKEBASE_CONFIG = {
# MAGIC     "instance_name": "Lakebase",
# MAGIC     "conn_host": "instance-6fbb9a7e-4a68-47e5-9b92-19b181e35f40.database.azuredatabricks.net",
# MAGIC     "conn_db_name": "databricks_postgres",
# MAGIC     "conn_ssl_mode": "require",
# MAGIC     "conn_port": 5432,
# MAGIC }
# MAGIC
# MAGIC ###############################################################################
# MAGIC # Cache Lakebase (tabela simples com PK composta)
# MAGIC ###############################################################################
# MAGIC class LakebaseCache(BaseCache):
# MAGIC     def __init__(self, pool: ConnectionPool, table_name: str = "llm_cache"):
# MAGIC         self.pool = pool
# MAGIC         self.table_name = table_name
# MAGIC         self._create_table_if_not_exists()
# MAGIC
# MAGIC     def _create_table_if_not_exists(self):
# MAGIC         sql = f"""
# MAGIC         CREATE TABLE IF NOT EXISTS {self.table_name} (
# MAGIC           prompt TEXT,
# MAGIC           llm_string TEXT,
# MAGIC           return_val TEXT,
# MAGIC           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# MAGIC           PRIMARY KEY (prompt, llm_string)
# MAGIC         );
# MAGIC         """
# MAGIC         try:
# MAGIC             with self.pool.connection() as conn:
# MAGIC                 conn.execute(sql)
# MAGIC         except Exception as e:
# MAGIC             logger.warning(f"Erro ao verificar/criar tabela de cache: {e}")
# MAGIC
# MAGIC     def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
# MAGIC         sql = f"SELECT return_val FROM {self.table_name} WHERE prompt = %s AND llm_string = %s"
# MAGIC         try:
# MAGIC             with self.pool.connection() as conn:
# MAGIC                 res = conn.execute(sql, (prompt, llm_string)).fetchone()
# MAGIC                 if res:
# MAGIC                     val = res["return_val"] if isinstance(res, dict) else res[0]
# MAGIC                     generations_dict = json.loads(val)
# MAGIC                     return [Generation(**gen) for gen in generations_dict]
# MAGIC         except Exception as e:
# MAGIC             logger.error(f"Cache lookup failed: {e}")
# MAGIC         return None
# MAGIC
# MAGIC     def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
# MAGIC         generations_dict = [gen.dict() for gen in return_val]
# MAGIC         json_val = json.dumps(generations_dict)
# MAGIC         sql = f"""
# MAGIC         INSERT INTO {self.table_name} (prompt, llm_string, return_val)
# MAGIC         VALUES (%s, %s, %s)
# MAGIC         ON CONFLICT (prompt, llm_string) DO NOTHING
# MAGIC         """
# MAGIC         try:
# MAGIC             with self.pool.connection() as conn:
# MAGIC                 conn.execute(sql, (prompt, llm_string, json_val))
# MAGIC         except Exception as e:
# MAGIC             logger.error(f"Cache update failed: {e}")
# MAGIC
# MAGIC     def clear(self) -> None:
# MAGIC         """Limpa toda a tabela de cache."""
# MAGIC         sql = f"DELETE FROM {self.table_name}"
# MAGIC         try:
# MAGIC             with self.pool.connection() as conn:
# MAGIC                 conn.execute(sql)
# MAGIC         except Exception as e:
# MAGIC             logger.error(f"Cache clear failed: {e}")
# MAGIC
# MAGIC ###############################################################################
# MAGIC # Ferramentas do agente
# MAGIC ###############################################################################
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC     last_result: Optional[str]
# MAGIC
# MAGIC def duckduckgo_research_summary(state: AgentState):
# MAGIC     """Busca ampla na web via DuckDuckGo, extrai conteúdo principal das páginas e gera um resumo em PT-BR com fontes."""
# MAGIC     import requests
# MAGIC     from bs4 import BeautifulSoup
# MAGIC
# MAGIC     # 1) Extrair consulta
# MAGIC     query = "Databricks Lakebase"
# MAGIC     try:
# MAGIC         for msg in reversed(state.get("messages", [])):
# MAGIC             md = msg.model_dump()
# MAGIC             if md.get("type") == "human" and md.get("content"):
# MAGIC                 q = str(md["content"]).strip()
# MAGIC                 if q:
# MAGIC                     query = q
# MAGIC                 break
# MAGIC     except Exception:
# MAGIC         pass
# MAGIC
# MAGIC     # 2) Buscar links no DuckDuckGo
# MAGIC     ddg_url = "https://html.duckduckgo.com/html/"
# MAGIC     headers = {
# MAGIC         "User-Agent": (
# MAGIC             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
# MAGIC             "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# MAGIC         )
# MAGIC     }
# MAGIC     links = []
# MAGIC     try:
# MAGIC         r = requests.get(ddg_url, params={"q": query}, headers=headers, timeout=10)
# MAGIC         r.raise_for_status()
# MAGIC         soup = BeautifulSoup(r.text, "html.parser")
# MAGIC         for a in soup.select("a.result__a")[:5]:
# MAGIC             href = a.get("href", "")
# MAGIC             resolved = href
# MAGIC             # Extrair URL real do redirecionamento /l/?uddg=
# MAGIC             try:
# MAGIC                 parsed = urlparse(href)
# MAGIC                 qs = parse_qs(parsed.query or "")
# MAGIC                 if "uddg" in qs and qs["uddg"]:
# MAGIC                     resolved = unquote(qs["uddg"][0])
# MAGIC             except Exception:
# MAGIC                 pass
# MAGIC             title = a.get_text(strip=True)
# MAGIC             links.append({"title": title, "url": resolved})
# MAGIC     except Exception as e:
# MAGIC         links = []
# MAGIC         logger.warning(f"DuckDuckGo falhou: {e}")
# MAGIC
# MAGIC     # 3) Extrair texto principal
# MAGIC     def extract_main_text(html: str) -> str:
# MAGIC         soup = BeautifulSoup(html, "html.parser")
# MAGIC         for tag in ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]:
# MAGIC             for t in soup.find_all(tag):
# MAGIC                 t.decompose()
# MAGIC         headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
# MAGIC         paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
# MAGIC         text = "\n".join(headings + paras)
# MAGIC         return text[:4000] if text else ""
# MAGIC
# MAGIC     corpus = []
# MAGIC     sources = []
# MAGIC     for lk in links:
# MAGIC         url = lk["url"]
# MAGIC         try:
# MAGIC             resp = requests.get(url, headers=headers, timeout=10)
# MAGIC             resp.raise_for_status()
# MAGIC             txt = extract_main_text(resp.text)
# MAGIC             if txt:
# MAGIC                 corpus.append(f"Fonte: {lk['title']} — {url}\n\n{txt}")
# MAGIC                 sources.append(f"- {lk['title']} — {url}")
# MAGIC         except Exception:
# MAGIC             continue
# MAGIC
# MAGIC     if not corpus:
# MAGIC         text = f"Não consegui obter conteúdo suficiente para '{query}'. Tente refinar a consulta."
# MAGIC         state["messages"].append(AIMessage(content=text))
# MAGIC         state["last_result"] = text
# MAGIC         return state
# MAGIC
# MAGIC     # 4) Síntese com ChatDatabricks (PT-BR)
# MAGIC     llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC     prompt = (
# MAGIC         "Você é um pesquisador. Leia os textos a seguir (corpus) e produza um resumo conciso em português, "
# MAGIC         "com no máximo 7 bullets, focando pontos‑chave, números e definições. "
# MAGIC         "Evite redundância. Não invente fatos não presentes. "
# MAGIC         "Corpus:\n\n" + "\n\n---\n\n".join(corpus)
# MAGIC     )
# MAGIC     try:
# MAGIC         resp = llm.invoke(prompt)
# MAGIC         summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
# MAGIC     except Exception:
# MAGIC         summary = "\n".join(corpus[:2])[:1500]
# MAGIC         summary = f"(Fallback de síntese)\n\n{summary}"
# MAGIC
# MAGIC     result = f"{summary}\n\nFontes:\n" + "\n".join(sources)
# MAGIC     state["messages"].append(AIMessage(content=result))
# MAGIC     state["last_result"] = result
# MAGIC     return state
# MAGIC
# MAGIC def describe_image_from_url(state: AgentState):
# MAGIC     """Recebe uma URL de imagem, converte em base64 e envia ao endpoint multimodal via ChatDatabricks; retorna só a descrição em PT-BR."""
# MAGIC     import requests
# MAGIC     media_type = "image/jpeg"
# MAGIC
# MAGIC     # 1) Obter URL (custom_inputs ou última mensagem humana)
# MAGIC     url = None
# MAGIC     ci = state.get("custom_inputs") or {}
# MAGIC     if isinstance(ci, dict) and ci.get("url"):
# MAGIC         url = str(ci["url"]).strip()
# MAGIC     if not url:
# MAGIC         try:
# MAGIC             for msg in reversed(state.get("messages", [])):
# MAGIC                 md = msg.model_dump()
# MAGIC                 if md.get("type") == "human" and md.get("content"):
# MAGIC                     candidate = str(md["content"]).strip()
# MAGIC                     if candidate.lower().startswith(("http://", "https://")):
# MAGIC                         url = candidate
# MAGIC                         break
# MAGIC         except Exception:
# MAGIC             pass
# MAGIC     if not url:
# MAGIC         text = "Por favor, forneça uma URL de imagem (ex.: https://upload.wikimedia.org/.../imagem.jpg)."
# MAGIC         state["messages"].append(AIMessage(content=text))
# MAGIC         state["last_result"] = text
# MAGIC         return state
# MAGIC
# MAGIC     # 2) Baixar imagem e converter para base64 (com resize opcional para reduzir payload)
# MAGIC     headers = {"User-Agent": "Image-Describe-Agent/1.0"}
# MAGIC     try:
# MAGIC         resp = requests.get(url, headers=headers, timeout=15)
# MAGIC         resp.raise_for_status()
# MAGIC         ctype = resp.headers.get("Content-Type", "")
# MAGIC         if "image/" in ctype:
# MAGIC             media_type = ctype.split(";")[0].strip()
# MAGIC         img_bytes = resp.content
# MAGIC
# MAGIC         # Reduz tamanho com Pillow se disponível
# MAGIC         try:
# MAGIC             from PIL import Image
# MAGIC             img = Image.open(BytesIO(img_bytes)).convert("RGB")
# MAGIC             max_w = 1280
# MAGIC             if img.width > max_w:
# MAGIC                 new_h = int(img.height * (max_w / img.width))
# MAGIC                 img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
# MAGIC             buf = BytesIO()
# MAGIC             img.save(buf, format="JPEG", quality=85)
# MAGIC             img_b64 = base64_encode(buf.getvalue())
# MAGIC             media_type = "image/jpeg"
# MAGIC         except Exception:
# MAGIC             # fallback sem Pillow
# MAGIC             img_b64 = base64_encode(img_bytes)
# MAGIC             if media_type not in ("image/jpeg", "image/png", "image/webp"):
# MAGIC                 media_type = "image/jpeg"
# MAGIC     except Exception as e:
# MAGIC         text = f"Falha ao baixar a imagem: {e}"
# MAGIC         state["messages"].append(AIMessage(content=text))
# MAGIC         state["last_result"] = text
# MAGIC         return state
# MAGIC
# MAGIC     # 3) Enviar ao endpoint multimodal via ChatDatabricks (LangChain)
# MAGIC     try:
# MAGIC         llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC         content_blocks = [
# MAGIC             {"type": "text", "text": "Descreva a imagem em detalhes em português."},
# MAGIC             {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
# MAGIC         ]
# MAGIC         # Passa como HumanMessage multimodal
# MAGIC         resp = llm.invoke([HumanMessage(content=content_blocks)])
# MAGIC         description = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
# MAGIC     except Exception as e:
# MAGIC         text = f"Falha ao descrever a imagem no endpoint: {e}"
# MAGIC         state["messages"].append(AIMessage(content=text))
# MAGIC         state["last_result"] = text
# MAGIC         return state
# MAGIC
# MAGIC     # 4) Retornar apenas a descrição (sem base64)
# MAGIC     state["messages"].append(AIMessage(content=description))
# MAGIC     state["last_result"] = description
# MAGIC     return state
# MAGIC
# MAGIC def base64_encode(data: bytes) -> str:
# MAGIC     import base64
# MAGIC     return base64.b64encode(data).decode("utf-8")
# MAGIC
# MAGIC # Registrar exatamente 2 ferramentas
# MAGIC tools = [duckduckgo_research_summary, describe_image_from_url]
# MAGIC
# MAGIC # (Opcional) UC function toolkit
# MAGIC UC_TOOL_NAMES: list[str] = []
# MAGIC if UC_TOOL_NAMES:
# MAGIC     uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC     tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # (Opcional) vector search tools
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC tools.extend(VECTOR_SEARCH_TOOLS)
# MAGIC
# MAGIC ###############################################################################
# MAGIC # Conexão PG com credencial Lakebase rotativa + cache
# MAGIC ###############################################################################
# MAGIC class CredentialConnection(psycopg.Connection):
# MAGIC     """Custom connection que injeta credencial de Lakebase com cache."""
# MAGIC     workspace_client = None
# MAGIC     instance_name = None
# MAGIC     _cached_credential = None
# MAGIC     _cache_timestamp = None
# MAGIC     _cache_duration = 3000  # segundos (padrão 50 min)
# MAGIC     _cache_lock = Lock()
# MAGIC
# MAGIC     @classmethod
# MAGIC     def connect(cls, conninfo="", **kwargs):
# MAGIC         if cls.workspace_client is None or cls.instance_name is None:
# MAGIC             raise ValueError("workspace_client e instance_name precisam estar configurados.")
# MAGIC         kwargs["password"] = cls._get_cached_credential()
# MAGIC         return super().connect(conninfo, **kwargs)
# MAGIC
# MAGIC     @classmethod
# MAGIC     def _get_cached_credential(cls):
# MAGIC         with cls._cache_lock:
# MAGIC             now = time.time()
# MAGIC             if (
# MAGIC                 cls._cached_credential
# MAGIC                 and cls._cache_timestamp
# MAGIC                 and (now - cls._cache_timestamp < cls._cache_duration)
# MAGIC             ):
# MAGIC                 return cls._cached_credential
# MAGIC
# MAGIC             cred = cls.workspace_client.database.generate_database_credential(
# MAGIC                 request_id=str(uuid.uuid4()),
# MAGIC                 instance_names=[cls.instance_name],
# MAGIC             )
# MAGIC             cls._cached_credential = cred.token
# MAGIC             cls._cache_timestamp = now
# MAGIC             return cls._cached_credential
# MAGIC
# MAGIC ###############################################################################
# MAGIC # Agente com LangGraph + checkpointing em Lakebase
# MAGIC ###############################################################################
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     """Agente stateful com Lakebase PostgreSQL checkpointing e cache."""
# MAGIC
# MAGIC     def __init__(self, lakebase_config: dict[str, Any]):
# MAGIC         self.lakebase_config = lakebase_config
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC
# MAGIC         # LLM e ferramentas
# MAGIC         self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC         self.system_prompt = SYSTEM_PROMPT
# MAGIC         self.model_with_tools = self.model.bind_tools(tools) if tools else self.model
# MAGIC
# MAGIC         # Configuração de pool
# MAGIC         self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
# MAGIC         self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
# MAGIC         self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
# MAGIC         cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
# MAGIC         CredentialConnection._cache_duration = cache_duration_minutes * 60
# MAGIC
# MAGIC         # Inicializa o pool com credenciais rotativas
# MAGIC         self._connection_pool = self._create_rotating_pool()
# MAGIC
# MAGIC         # MLflow autolog: apenas tracing para LangChain (evita warnings de Spark)
# MAGIC         mlflow.langchain.autolog(log_traces=True, silent=True)
# MAGIC
# MAGIC         # Cache no Lakebase
# MAGIC         self._setup_lakebase_cache()
# MAGIC
# MAGIC     def _get_username(self) -> str:
# MAGIC         """Usuário de conexão: SP (application_id) ou e‑mail do usuário."""
# MAGIC         try:
# MAGIC             sp = self.workspace_client.current_service_principal.me()
# MAGIC             return sp.application_id
# MAGIC         except Exception:
# MAGIC             return self.workspace_client.current_user.me().user_name
# MAGIC
# MAGIC     def _setup_lakebase_cache(self):
# MAGIC         try:
# MAGIC             cache = LakebaseCache(pool=self._connection_pool, table_name="llm_exact_cache")
# MAGIC             set_llm_cache(cache)
# MAGIC             logger.info("✅ LakebaseCache configurado com sucesso.")
# MAGIC         except Exception as e:
# MAGIC             logger.error(f"❌ Falha ao configurar LakebaseCache: {e}")
# MAGIC
# MAGIC     def _create_rotating_pool(self) -> ConnectionPool:
# MAGIC         CredentialConnection.workspace_client = self.workspace_client
# MAGIC         CredentialConnection.instance_name = self.lakebase_config["instance_name"]
# MAGIC
# MAGIC         username = self._get_username()
# MAGIC         host = self.lakebase_config["conn_host"]
# MAGIC         database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
# MAGIC         sslmode = self.lakebase_config.get("conn_ssl_mode", "require")
# MAGIC         port = int(self.lakebase_config.get("conn_port", 5432))
# MAGIC
# MAGIC         conninfo = f"dbname={database} user={username} host={host} port={port} sslmode={sslmode}"
# MAGIC
# MAGIC         pool = ConnectionPool(
# MAGIC             conninfo=conninfo,
# MAGIC             connection_class=CredentialConnection,
# MAGIC             min_size=self.pool_min_size,
# MAGIC             max_size=self.pool_max_size,
# MAGIC             timeout=self.pool_timeout,
# MAGIC             open=True,
# MAGIC             kwargs={
# MAGIC                 "autocommit": True,  # necessário para PostgresSaver.setup()
# MAGIC                 "row_factory": dict_row,  # PostgresSaver usa dict para rows
# MAGIC                 "keepalives": 1,
# MAGIC                 "keepalives_idle": 30,
# MAGIC                 "keepalives_interval": 10,
# MAGIC                 "keepalives_count": 5,
# MAGIC             },
# MAGIC         )
# MAGIC         try:
# MAGIC             with pool.connection() as conn:
# MAGIC                 # Sanity check
# MAGIC                 with conn.cursor() as cur:
# MAGIC                     cur.execute("SELECT 1")
# MAGIC                 # Garantir que as tabelas de checkpoint existem (primeira execução)
# MAGIC                 try:
# MAGIC                     checkpointer = PostgresSaver(conn)
# MAGIC                     checkpointer.setup()
# MAGIC                 except Exception as e:
# MAGIC                     logger.info(f"PostgresSaver.setup() não necessário ou já executado: {e}")
# MAGIC
# MAGIC             logger.info(
# MAGIC                 f"Pool criado (min={self.pool_min_size}, max={self.pool_max_size}, "
# MAGIC                 f"token_cache={CredentialConnection._cache_duration/60:.0f} min)"
# MAGIC             )
# MAGIC         except Exception as e:
# MAGIC             pool.close()
# MAGIC             raise ConnectionError(f"Falha ao criar pool de conexão: {e}")
# MAGIC         return pool
# MAGIC
# MAGIC     @contextmanager
# MAGIC     def get_connection(self):
# MAGIC         with self._connection_pool.connection() as conn:
# MAGIC             yield conn
# MAGIC
# MAGIC     def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
# MAGIC         responses = []
# MAGIC         for message in messages:
# MAGIC             md = message.model_dump()
# MAGIC             t = md["type"]
# MAGIC             if t == "ai":
# MAGIC                 if tool_calls := md.get("tool_calls"):
# MAGIC                     for tc in tool_calls:
# MAGIC                         responses.append(
# MAGIC                             self.create_function_call_item(
# MAGIC                                 id=md.get("id") or str(uuid.uuid4()),
# MAGIC                                 call_id=tc["id"],
# MAGIC                                 name=tc["name"],
# MAGIC                                 arguments=json.dumps(tc["args"]),
# MAGIC                             )
# MAGIC                         )
# MAGIC                 else:
# MAGIC                     responses.append(
# MAGIC                         self.create_text_output_item(
# MAGIC                             text=md.get("content", ""),
# MAGIC                             id=md.get("id") or str(uuid.uuid4()),
# MAGIC                         )
# MAGIC                     )
# MAGIC             elif t == "tool":
# MAGIC                 responses.append(
# MAGIC                     self.create_function_call_output_item(
# MAGIC                         call_id=md.get("tool_call_id", str(uuid.uuid4())),
# MAGIC                         output=str(md["content"]),
# MAGIC                     )
# MAGIC                 )
# MAGIC             elif t == "human":
# MAGIC                 responses.append({"role": "user", "content": md.get("content", "")})
# MAGIC         return responses
# MAGIC
# MAGIC     def _create_graph(self, checkpointer: Optional[PostgresSaver], use_tools: bool = True):
# MAGIC         """Cria o workflow; aceita checkpointer=None e usa ferramentas conforme use_tools."""
# MAGIC         def should_continue(state: AgentState):
# MAGIC             last = state["messages"][-1]
# MAGIC             return "continue" if isinstance(last, AIMessage) and last.tool_calls else "end"
# MAGIC
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC         # Seleciona modelo com/sem ferramentas
# MAGIC         model_base = self.model_with_tools if use_tools and tools else self.model
# MAGIC         model_runnable = preprocessor | model_base
# MAGIC
# MAGIC         def call_model(state: AgentState, config):
# MAGIC             response = model_runnable.invoke(state, config)
# MAGIC             return {"messages": [response]}
# MAGIC
# MAGIC         workflow = StateGraph(AgentState)
# MAGIC         workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC
# MAGIC         if use_tools and tools:
# MAGIC             workflow.add_node("tools", ToolNode(tools))
# MAGIC             workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
# MAGIC             workflow.add_edge("tools", "agent")
# MAGIC         else:
# MAGIC             workflow.add_edge("agent", END)
# MAGIC
# MAGIC         workflow.set_entry_point("agent")
# MAGIC         return workflow.compile(checkpointer=checkpointer)
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         if "thread_id" not in ci:
# MAGIC             ci["thread_id"] = str(uuid.uuid4())
# MAGIC         request.custom_inputs = ci
# MAGIC
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs={"thread_id": ci["thread_id"]})
# MAGIC
# MAGIC     def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         custom_inputs = request.custom_inputs or {}
# MAGIC         thread_id = custom_inputs.get("thread_id", str(uuid.uuid4()))
# MAGIC         # Permitir desligar ferramentas na validação: custom_inputs.tool_choice="none"
# MAGIC         tool_choice = (custom_inputs.get("tool_choice") or "auto").lower()
# MAGIC         use_tools = tool_choice != "none"
# MAGIC
# MAGIC         # Fallback via variável de ambiente (para validação do MLflow)
# MAGIC         env_tool_choice = (os.getenv("AGENT_TOOL_CHOICE") or "").lower()
# MAGIC         if env_tool_choice == "none":
# MAGIC             use_tools = False
# MAGIC
# MAGIC         # Converte mensagens Responses -> formato CC para LangChain
# MAGIC         cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
# MAGIC         langchain_msgs = cc_msgs
# MAGIC
# MAGIC         checkpoint_config = {"configurable": {"thread_id": thread_id}}
# MAGIC
# MAGIC         with self.get_connection() as conn:
# MAGIC             checkpointer = PostgresSaver(conn)
# MAGIC             graph = self._create_graph(checkpointer, use_tools=use_tools)
# MAGIC
# MAGIC             for event in graph.stream({"messages": langchain_msgs}, checkpoint_config, stream_mode=["updates", "messages"]):
# MAGIC                 if event[0] == "updates":
# MAGIC                     for node_data in event[1].values():
# MAGIC                         msgs = node_data["messages"] if isinstance(node_data["messages"], list) else [node_data["messages"]]
# MAGIC                         for item in self._langchain_to_responses(msgs):
# MAGIC                             yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
# MAGIC                 elif event[0] == "messages":
# MAGIC                     try:
# MAGIC                         chunk = event[1][0]
# MAGIC                         if isinstance(chunk, AIMessageChunk) and chunk.content:
# MAGIC                             yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=chunk.content, item_id=chunk.id))
# MAGIC                     except Exception as e:
# MAGIC                         logger.error(f"Erro no streaming de chunk: {e}")
# MAGIC
# MAGIC # ----- Export model -----
# MAGIC AGENT = LangGraphResponsesAgent(LAKEBASE_CONFIG)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# Reiniciar sys
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Test the Agent locally

# COMMAND ----------

from agentlakebase import LangGraphResponsesAgent, LAKEBASE_CONFIG

agent = LangGraphResponsesAgent(LAKEBASE_CONFIG)

# Exibir o draw_mermaid
from IPython.display import Image, display

display(Image(agent._create_graph(checkpointer=None).get_graph().draw_mermaid_png()))

# COMMAND ----------

from agentlakebase import AGENT
# Message 1, don't include thread_id (creates new thread)
result = AGENT.predict({
    "input": [{"role": "user", "content": "oi, tudo bem?"}]
})
print(result.model_dump(exclude_none=True))
thread_id = result.custom_outputs["thread_id"]

# COMMAND ----------

# Message 2, include thread ID and notice how agent remembers context from previous predict message
response2 = AGENT.predict({
    "input": [{"role": "user", "content": "qual foi a primeira coisa que eu te disse?"}],
    "custom_inputs": {"thread_id": thread_id}
})
print("Response 2:", response2.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC # Log the agent as an MLflow model
# MAGIC

# COMMAND ----------

import os, uuid
import mlflow
from pkg_resources import get_distribution
from agentlakebase import tools, LLM_ENDPOINT_NAME, LAKEBASE_CONFIG
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksLakebase
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

# Desliga ferramentas na validação por fallback de ambiente (agente já suporta)
os.environ["AGENT_TOOL_CHOICE"] = "none"

resources = [
    DatabricksServingEndpoint(LLM_ENDPOINT_NAME),
    DatabricksLakebase(database_instance_name=LAKEBASE_CONFIG["instance_name"]),
]

for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

# THREAD NOVA A CADA LOG para evitar memória anterior com tool_use
input_example = {
    "input": [{"role": "user", "content": "Responda apenas 'OK'."}],
    "custom_inputs": {
        "thread_id": str(uuid.uuid4()),
        "tool_choice": "none",  # agente também lê isso, se repassado
    },
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agentlakebase",
        python_model="agentlakebase.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            "databricks-langchain",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"langgraph-checkpoint-postgres=={get_distribution('langgraph-checkpoint-postgres').version}",
            "psycopg[binary,pool]",
            f"pydantic=={get_distribution('pydantic').version}",
            "bs4",
        ],
    )

# Limpa a variável de ambiente após o log, para não afetar execuções normais
os.environ.pop("AGENT_TOOL_CHOICE", None)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the agent with Agent Evaluation
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Correctness, Safety
from typing import Any

eval_dataset = [
    {
        "inputs": {
            "input": [
                {"role": "user", "content": "O que é a Databricks??"}
            ]
        },
        "expectations": {
            "expected_response": "A Databricks é uma empresa de dados e IA."
        }
    }
]

def _normalize_response(resp: Any) -> str:
    # 1) Se já for string
    if isinstance(resp, str):
        return resp
    # 2) Tente atributos comuns
    for attr in ("output", "response", "text", "content", "message", "answer"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            # Alguns frameworks têm resp.response.output_text
            if hasattr(val, "output_text") and isinstance(val.output_text, str):
                return val.output_text
            if isinstance(val, str):
                return val
    # 3) Tente serializar para dict (pydantic/dataclass)
    try:
        d = (
            resp.model_dump() if hasattr(resp, "model_dump")
            else resp.dict() if hasattr(resp, "dict")
            else getattr(resp, "__dict__", None)
        )
        if isinstance(d, dict):
            # tente chaves comuns
            for key in ("output", "response", "text", "content", "message", "answer"):
                val = d.get(key)
                if isinstance(val, str):
                    return val
                # suporte resp["response"]["output_text"]
                if isinstance(val, dict) and isinstance(val.get("output_text"), str):
                    return val["output_text"]
            # fallback: primeira string
            for v in d.values():
                if isinstance(v, str):
                    return v
    except Exception:
        pass
    # 4) Fallback: representação textual (melhor do que cair de volta no input)
    return str(resp)

def predict_fn(input):  # casa com "input" no dataset
    resp = AGENT.predict({"input": input})
    response_text = _normalize_response(resp)
    return {
        "trace": {
            "input": input,
            "model": "AGENT-v1",  # opcional
            "outputs": [response_text],  # outputs deve ser lista
        }
    }

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[Correctness(), Safety()],
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API.

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agentlakebase",
    input_data={"input": [{"role": "user", "content": "o que é a Databricks?"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model to Unity Catalog
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "catalogo_databricks"
schema = "agentes_ia"
model_name = "agentlakebase"

UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"agentlakebase": "Yes"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Usar Endpoint Deployado

# COMMAND ----------

from openai import OpenAI
import os
import getpass
import uuid

client = OpenAI(
    api_key=getpass.getpass("Digite seu token: "),
    base_url="https://adb-3250511655996160.0.azuredatabricks.net/serving-endpoints/"
)

thread_id = str(uuid.uuid4())

response1 = client.responses.create(
    model="agents_catalogo_databricks-agentes_ia-agentlakebase",
    input=[{"role": "user", "content": "Olá, meu nome é João. O que tem nessa url https://www.databricks.com/sites/default/files/inline-images/building-data-pipelines-with-delta-lake-120823.png?"}],
    extra_body={ 
        "custom_inputs": {"thread_id": thread_id}
    }
)
response2 = client.responses.create(
    model="agents_catalogo_databricks-agentes_ia-agentlakebase",
    input=[{"role": "user", "content": "Qual foi a última coisa que te disse?"}],
    extra_body={
        "custom_inputs": {"thread_id": thread_id}
    }
)

print(response1)
print(response2)