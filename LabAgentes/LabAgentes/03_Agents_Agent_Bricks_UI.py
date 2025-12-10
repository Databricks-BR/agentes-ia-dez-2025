# Databricks notebook source
# MAGIC %md
# MAGIC ![](./Images/image.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Laboratório de Agentes com GenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação da Genie

# COMMAND ----------

# MAGIC %md
# MAGIC ![Genie.png](./Images/Genie.png "Genie.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação do Knowledge Assistant

# COMMAND ----------

# Create volume if not exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS catalogo_databricks.agentes_ia.volumepdf")

# COMMAND ----------

src = "dbfs:/Workspace/Users/daniel.baraldi@databricks.com/LabAgentes/AttentionAllYouNeed.pdf"
dst = "/Volumes/catalogo_databricks/agentes_ia/volumepdf"
dbutils.fs.cp(src, dst, recurse=False)  # ou dbutils.fs.mv para mover

# COMMAND ----------

#Read pdf file from and use ai_parse_document to extract
from pyspark.sql.functions import expr
df=spark.read.format("binaryFile").load("/Volumes/catalogo_databricks/agentes_ia/volumepdf").withColumn(


    "parsed",
    expr("ai_parse_document(content)"))
display(df)

# COMMAND ----------

#Extract all columns from dataframe
from pyspark.sql.functions import col, parse_json

df_copy=df.withColumn(
   "parsed_json",
   parse_json(col("parsed").cast("string"))) \
 .selectExpr(
   "path",
   "parsed_json:document:elements")
display(df_copy)

# COMMAND ----------

#Explode into multiple rows with content from each page
from pyspark.sql.functions import explode
from pyspark.sql.functions import from_json, explode, col


# Define the expected array type for your data
from pyspark.sql.types import ArrayType, StringType
array_schema = ArrayType(StringType())


# Convert 'elements' (VARIANT) to array by parsing as JSON string
df_copy2 = df_copy.withColumn("elements_array", from_json(col("elements").cast("string"), array_schema))

# Explode the new array column
df_copy3 = df_copy2.select("path", explode(col("elements_array")).alias("element"))
display(df_copy3)

# COMMAND ----------

display(df_copy3)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

#Add id to the dataframe
df_copy4=df_copy3.withColumn("id", monotonically_increasing_id())
display(df_copy4)

# COMMAND ----------

#Write dataframe to table users.daniel_baraldi.documents
df_copy4.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"catalogo_databricks.agentes_ia.documentos")

# COMMAND ----------

# Atualize libs necessárias no notebook
%pip install -U typing_extensions==4.12.2 pydantic>=2.6 pydantic-core>=2.20 databricks-mcp

# Reinicie o interpretador do notebook para carregar as libs atualizadas
dbutils.library.restartPython()

# COMMAND ----------

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
mcp = DatabricksMCPClient(server_url="https://mcp-hello-3250511655996160.0.azure.databricksapps.com/mcp", workspace_client=WorkspaceClient())
print([t.name for t in mcp.list_tools()])

# COMMAND ----------

# Crie o índice vetorial via UI

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação do Agente Orquestrador com Agent Bricks

# COMMAND ----------

# MAGIC %md
# MAGIC ![agentbricks.png](./Images/agentbricks.png "agentbricks.png")