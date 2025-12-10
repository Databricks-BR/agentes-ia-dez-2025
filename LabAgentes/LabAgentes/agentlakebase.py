import json
import logging
import os
import time
import uuid
from threading import Lock
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, List
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs, unquote, quote
from io import BytesIO

import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# Cache core
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation
from langchain_core.globals import set_llm_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

############################################
# LLM endpoint e system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools to answer questions."

############################################
# LAKEBASE CONFIG
############################################
LAKEBASE_CONFIG = {
    "instance_name": "Lakebase",
    "conn_host": "instance-6fbb9a7e-4a68-47e5-9b92-19b181e35f40.database.azuredatabricks.net",
    "conn_db_name": "databricks_postgres",
    "conn_ssl_mode": "require",
    "conn_port": 5432,
}

###############################################################################
# Cache Lakebase (tabela simples com PK composta)
###############################################################################
class LakebaseCache(BaseCache):
    def __init__(self, pool: ConnectionPool, table_name: str = "llm_cache"):
        self.pool = pool
        self.table_name = table_name
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
          prompt TEXT,
          llm_string TEXT,
          return_val TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (prompt, llm_string)
        );
        """
        try:
            with self.pool.connection() as conn:
                conn.execute(sql)
        except Exception as e:
            logger.warning(f"Erro ao verificar/criar tabela de cache: {e}")

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        sql = f"SELECT return_val FROM {self.table_name} WHERE prompt = %s AND llm_string = %s"
        try:
            with self.pool.connection() as conn:
                res = conn.execute(sql, (prompt, llm_string)).fetchone()
                if res:
                    val = res["return_val"] if isinstance(res, dict) else res[0]
                    generations_dict = json.loads(val)
                    return [Generation(**gen) for gen in generations_dict]
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
        return None

    def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
        generations_dict = [gen.dict() for gen in return_val]
        json_val = json.dumps(generations_dict)
        sql = f"""
        INSERT INTO {self.table_name} (prompt, llm_string, return_val)
        VALUES (%s, %s, %s)
        ON CONFLICT (prompt, llm_string) DO NOTHING
        """
        try:
            with self.pool.connection() as conn:
                conn.execute(sql, (prompt, llm_string, json_val))
        except Exception as e:
            logger.error(f"Cache update failed: {e}")

    def clear(self) -> None:
        """Limpa toda a tabela de cache."""
        sql = f"DELETE FROM {self.table_name}"
        try:
            with self.pool.connection() as conn:
                conn.execute(sql)
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

###############################################################################
# Ferramentas do agente
###############################################################################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    last_result: Optional[str]

def duckduckgo_research_summary(state: AgentState):
    """Busca ampla na web via DuckDuckGo, extrai conteúdo principal das páginas e gera um resumo em PT-BR com fontes."""
    import requests
    from bs4 import BeautifulSoup

    # 1) Extrair consulta
    query = "Databricks Lakebase"
    try:
        for msg in reversed(state.get("messages", [])):
            md = msg.model_dump()
            if md.get("type") == "human" and md.get("content"):
                q = str(md["content"]).strip()
                if q:
                    query = q
                break
    except Exception:
        pass

    # 2) Buscar links no DuckDuckGo
    ddg_url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    links = []
    try:
        r = requests.get(ddg_url, params={"q": query}, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.result__a")[:5]:
            href = a.get("href", "")
            resolved = href
            # Extrair URL real do redirecionamento /l/?uddg=
            try:
                parsed = urlparse(href)
                qs = parse_qs(parsed.query or "")
                if "uddg" in qs and qs["uddg"]:
                    resolved = unquote(qs["uddg"][0])
            except Exception:
                pass
            title = a.get_text(strip=True)
            links.append({"title": title, "url": resolved})
    except Exception as e:
        links = []
        logger.warning(f"DuckDuckGo falhou: {e}")

    # 3) Extrair texto principal
    def extract_main_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]:
            for t in soup.find_all(tag):
                t.decompose()
        headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
        paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
        text = "\n".join(headings + paras)
        return text[:4000] if text else ""

    corpus = []
    sources = []
    for lk in links:
        url = lk["url"]
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            txt = extract_main_text(resp.text)
            if txt:
                corpus.append(f"Fonte: {lk['title']} — {url}\n\n{txt}")
                sources.append(f"- {lk['title']} — {url}")
        except Exception:
            continue

    if not corpus:
        text = f"Não consegui obter conteúdo suficiente para '{query}'. Tente refinar a consulta."
        state["messages"].append(AIMessage(content=text))
        state["last_result"] = text
        return state

    # 4) Síntese com ChatDatabricks (PT-BR)
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
    prompt = (
        "Você é um pesquisador. Leia os textos a seguir (corpus) e produza um resumo conciso em português, "
        "com no máximo 7 bullets, focando pontos‑chave, números e definições. "
        "Evite redundância. Não invente fatos não presentes. "
        "Corpus:\n\n" + "\n\n---\n\n".join(corpus)
    )
    try:
        resp = llm.invoke(prompt)
        summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    except Exception:
        summary = "\n".join(corpus[:2])[:1500]
        summary = f"(Fallback de síntese)\n\n{summary}"

    result = f"{summary}\n\nFontes:\n" + "\n".join(sources)
    state["messages"].append(AIMessage(content=result))
    state["last_result"] = result
    return state

def describe_image_from_url(state: AgentState):
    """Recebe uma URL de imagem, converte em base64 e envia ao endpoint multimodal via ChatDatabricks; retorna só a descrição em PT-BR."""
    import requests
    media_type = "image/jpeg"

    # 1) Obter URL (custom_inputs ou última mensagem humana)
    url = None
    ci = state.get("custom_inputs") or {}
    if isinstance(ci, dict) and ci.get("url"):
        url = str(ci["url"]).strip()
    if not url:
        try:
            for msg in reversed(state.get("messages", [])):
                md = msg.model_dump()
                if md.get("type") == "human" and md.get("content"):
                    candidate = str(md["content"]).strip()
                    if candidate.lower().startswith(("http://", "https://")):
                        url = candidate
                        break
        except Exception:
            pass
    if not url:
        text = "Por favor, forneça uma URL de imagem (ex.: https://upload.wikimedia.org/.../imagem.jpg)."
        state["messages"].append(AIMessage(content=text))
        state["last_result"] = text
        return state

    # 2) Baixar imagem e converter para base64 (com resize opcional para reduzir payload)
    headers = {"User-Agent": "Image-Describe-Agent/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image/" in ctype:
            media_type = ctype.split(";")[0].strip()
        img_bytes = resp.content

        # Reduz tamanho com Pillow se disponível
        try:
            from PIL import Image
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            max_w = 1280
            if img.width > max_w:
                new_h = int(img.height * (max_w / img.width))
                img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64_encode(buf.getvalue())
            media_type = "image/jpeg"
        except Exception:
            # fallback sem Pillow
            img_b64 = base64_encode(img_bytes)
            if media_type not in ("image/jpeg", "image/png", "image/webp"):
                media_type = "image/jpeg"
    except Exception as e:
        text = f"Falha ao baixar a imagem: {e}"
        state["messages"].append(AIMessage(content=text))
        state["last_result"] = text
        return state

    # 3) Enviar ao endpoint multimodal via ChatDatabricks (LangChain)
    try:
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        content_blocks = [
            {"type": "text", "text": "Descreva a imagem em detalhes em português."},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
        ]
        # Passa como HumanMessage multimodal
        resp = llm.invoke([HumanMessage(content=content_blocks)])
        description = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    except Exception as e:
        text = f"Falha ao descrever a imagem no endpoint: {e}"
        state["messages"].append(AIMessage(content=text))
        state["last_result"] = text
        return state

    # 4) Retornar apenas a descrição (sem base64)
    state["messages"].append(AIMessage(content=description))
    state["last_result"] = description
    return state

def base64_encode(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("utf-8")

# Registrar exatamente 2 ferramentas
tools = [duckduckgo_research_summary, describe_image_from_url]

# (Opcional) UC function toolkit
UC_TOOL_NAMES: list[str] = []
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    tools.extend(uc_toolkit.tools)

# (Opcional) vector search tools
VECTOR_SEARCH_TOOLS = []
tools.extend(VECTOR_SEARCH_TOOLS)

###############################################################################
# Conexão PG com credencial Lakebase rotativa + cache
###############################################################################
class CredentialConnection(psycopg.Connection):
    """Custom connection que injeta credencial de Lakebase com cache."""
    workspace_client = None
    instance_name = None
    _cached_credential = None
    _cache_timestamp = None
    _cache_duration = 3000  # segundos (padrão 50 min)
    _cache_lock = Lock()

    @classmethod
    def connect(cls, conninfo="", **kwargs):
        if cls.workspace_client is None or cls.instance_name is None:
            raise ValueError("workspace_client e instance_name precisam estar configurados.")
        kwargs["password"] = cls._get_cached_credential()
        return super().connect(conninfo, **kwargs)

    @classmethod
    def _get_cached_credential(cls):
        with cls._cache_lock:
            now = time.time()
            if (
                cls._cached_credential
                and cls._cache_timestamp
                and (now - cls._cache_timestamp < cls._cache_duration)
            ):
                return cls._cached_credential

            cred = cls.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[cls.instance_name],
            )
            cls._cached_credential = cred.token
            cls._cache_timestamp = now
            return cls._cached_credential

###############################################################################
# Agente com LangGraph + checkpointing em Lakebase
###############################################################################
class LangGraphResponsesAgent(ResponsesAgent):
    """Agente stateful com Lakebase PostgreSQL checkpointing e cache."""

    def __init__(self, lakebase_config: dict[str, Any]):
        self.lakebase_config = lakebase_config
        self.workspace_client = WorkspaceClient()

        # LLM e ferramentas
        self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        self.system_prompt = SYSTEM_PROMPT
        self.model_with_tools = self.model.bind_tools(tools) if tools else self.model

        # Configuração de pool
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
        self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
        cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
        CredentialConnection._cache_duration = cache_duration_minutes * 60

        # Inicializa o pool com credenciais rotativas
        self._connection_pool = self._create_rotating_pool()

        # MLflow autolog: apenas tracing para LangChain (evita warnings de Spark)
        mlflow.langchain.autolog(log_traces=True, silent=True)

        # Cache no Lakebase
        self._setup_lakebase_cache()

    def _get_username(self) -> str:
        """Usuário de conexão: SP (application_id) ou e‑mail do usuário."""
        try:
            sp = self.workspace_client.current_service_principal.me()
            return sp.application_id
        except Exception:
            return self.workspace_client.current_user.me().user_name

    def _setup_lakebase_cache(self):
        try:
            cache = LakebaseCache(pool=self._connection_pool, table_name="llm_exact_cache")
            set_llm_cache(cache)
            logger.info("✅ LakebaseCache configurado com sucesso.")
        except Exception as e:
            logger.error(f"❌ Falha ao configurar LakebaseCache: {e}")

    def _create_rotating_pool(self) -> ConnectionPool:
        CredentialConnection.workspace_client = self.workspace_client
        CredentialConnection.instance_name = self.lakebase_config["instance_name"]

        username = self._get_username()
        host = self.lakebase_config["conn_host"]
        database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
        sslmode = self.lakebase_config.get("conn_ssl_mode", "require")
        port = int(self.lakebase_config.get("conn_port", 5432))

        conninfo = f"dbname={database} user={username} host={host} port={port} sslmode={sslmode}"

        pool = ConnectionPool(
            conninfo=conninfo,
            connection_class=CredentialConnection,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            timeout=self.pool_timeout,
            open=True,
            kwargs={
                "autocommit": True,  # necessário para PostgresSaver.setup()
                "row_factory": dict_row,  # PostgresSaver usa dict para rows
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
        try:
            with pool.connection() as conn:
                # Sanity check
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                # Garantir que as tabelas de checkpoint existem (primeira execução)
                try:
                    checkpointer = PostgresSaver(conn)
                    checkpointer.setup()
                except Exception as e:
                    logger.info(f"PostgresSaver.setup() não necessário ou já executado: {e}")

            logger.info(
                f"Pool criado (min={self.pool_min_size}, max={self.pool_max_size}, "
                f"token_cache={CredentialConnection._cache_duration/60:.0f} min)"
            )
        except Exception as e:
            pool.close()
            raise ConnectionError(f"Falha ao criar pool de conexão: {e}")
        return pool

    @contextmanager
    def get_connection(self):
        with self._connection_pool.connection() as conn:
            yield conn

    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        responses = []
        for message in messages:
            md = message.model_dump()
            t = md["type"]
            if t == "ai":
                if tool_calls := md.get("tool_calls"):
                    for tc in tool_calls:
                        responses.append(
                            self.create_function_call_item(
                                id=md.get("id") or str(uuid.uuid4()),
                                call_id=tc["id"],
                                name=tc["name"],
                                arguments=json.dumps(tc["args"]),
                            )
                        )
                else:
                    responses.append(
                        self.create_text_output_item(
                            text=md.get("content", ""),
                            id=md.get("id") or str(uuid.uuid4()),
                        )
                    )
            elif t == "tool":
                responses.append(
                    self.create_function_call_output_item(
                        call_id=md.get("tool_call_id", str(uuid.uuid4())),
                        output=str(md["content"]),
                    )
                )
            elif t == "human":
                responses.append({"role": "user", "content": md.get("content", "")})
        return responses

    def _create_graph(self, checkpointer: Optional[PostgresSaver], use_tools: bool = True):
        """Cria o workflow; aceita checkpointer=None e usa ferramentas conforme use_tools."""
        def should_continue(state: AgentState):
            last = state["messages"][-1]
            return "continue" if isinstance(last, AIMessage) and last.tool_calls else "end"

        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
        )
        # Seleciona modelo com/sem ferramentas
        model_base = self.model_with_tools if use_tools and tools else self.model
        model_runnable = preprocessor | model_base

        def call_model(state: AgentState, config):
            response = model_runnable.invoke(state, config)
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", RunnableLambda(call_model))

        if use_tools and tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        workflow.set_entry_point("agent")
        return workflow.compile(checkpointer=checkpointer)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        ci = dict(request.custom_inputs or {})
        if "thread_id" not in ci:
            ci["thread_id"] = str(uuid.uuid4())
        request.custom_inputs = ci

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs={"thread_id": ci["thread_id"]})

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id", str(uuid.uuid4()))
        # Permitir desligar ferramentas na validação: custom_inputs.tool_choice="none"
        tool_choice = (custom_inputs.get("tool_choice") or "auto").lower()
        use_tools = tool_choice != "none"

        # Fallback via variável de ambiente (para validação do MLflow)
        env_tool_choice = (os.getenv("AGENT_TOOL_CHOICE") or "").lower()
        if env_tool_choice == "none":
            use_tools = False

        # Converte mensagens Responses -> formato CC para LangChain
        cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        langchain_msgs = cc_msgs

        checkpoint_config = {"configurable": {"thread_id": thread_id}}

        with self.get_connection() as conn:
            checkpointer = PostgresSaver(conn)
            graph = self._create_graph(checkpointer, use_tools=use_tools)

            for event in graph.stream({"messages": langchain_msgs}, checkpoint_config, stream_mode=["updates", "messages"]):
                if event[0] == "updates":
                    for node_data in event[1].values():
                        msgs = node_data["messages"] if isinstance(node_data["messages"], list) else [node_data["messages"]]
                        for item in self._langchain_to_responses(msgs):
                            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
                elif event[0] == "messages":
                    try:
                        chunk = event[1][0]
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            yield ResponsesAgentStreamEvent(**self.create_text_delta(delta=chunk.content, item_id=chunk.id))
                    except Exception as e:
                        logger.error(f"Erro no streaming de chunk: {e}")

# ----- Export model -----
AGENT = LangGraphResponsesAgent(LAKEBASE_CONFIG)
mlflow.models.set_model(AGENT)
