from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import langchain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import NGramOverlapExampleSelector
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX

llm = LlamaCpp(
    model_path="/Users/macpro/.cache/lm-studio/models/MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q8_0.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=10000,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
    #stop=[".\n\nQuestion"], # Stop generating just before the model would generate a new question
    repeat_penalty=1.25,
    temperature=0.0,
    max_tokens = 512
)

langchain.verbose = False

pg_uri = f"postgresql+psycopg2://user01:user01@192.168.89.149:5433/cdaplus"
db = SQLDatabase.from_uri(pg_uri,
        include_tables=['view_table_new'], # include only the tables you want to query. Reduces tokens.
        sample_rows_in_table_info=3,
        view_support = True
    )

cda_few_shot_prompt='''

You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer in Italian language to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

IGNORE "id" and "data_ann" fields when generating SQL queries.

Provide the final answer based on the SQL query result without inventing anything from sample rows.

When the generated query doesn't provide any result, reply "Non ho trovato nessun prodotto con tale corrispondenza." as the final answer. Here's an example:
Question: Elenca i prodotti di deposito, attivi, con categoria merce 'caffe'
SQLQuery:SELECT cod_prod, descr_mer , status , rifornibilita FROM view_table_new WHERE rifornibilita = 'C' AND status = ' ' AND descrizione_cat_mer ILIKE '%caffe%';
SQLResult:
Answer: Non ho trovato nessun prodotto con tale corrispondenza.

When the question referring to the date a product was created ('created_on' field) or updated ('updated_on' field), use the "YYYY-MM-DD" date format and return that specific field in the generated query.
For example, if you are asked about products created or updated on January 10, 2007, use the date "2007-01-10". Note that the time part of the date is not relevant for SQL queries and should be omitted.

When the question contains words enclosed in single quotes (' '), use the ILIKE operator with the '%' symbol to match similar patterns in the generated SQL query.
For example, if the question is "Quanti articoli con descrizione categoria statistica 'ARTICOLI 200X' ci sono?", the generated query should be "SELECT COUNT(*) FROM view_table_new WHERE descrizione_cat_sta ILIKE '%ARTICOLI 200X%';".

When the question refers to the products of a supplier enclosed in single quotes (' '), use the 'rag_soc' field and the ILIKE operator with the '%' symbol to match similar patterns in the generated SQL query to filter by the supplier's name.

When the question refers to a merchandise category enclosed in single quotes (' '), use the 'descrizione_cat_mer' field and the ILIKE operator with the '%' symbol to match similar patterns in the generated SQL query and filter by the specified merchandise category.

When the question contains the word 'descrizione', make sure to exclusively utilize the 'descr_mer' field for the query generation.

When the question contains the phrase 'categoria statistica', make sure to exclusively utilize the 'descrizione_cat_sta' field for the query generation.
When the question contains the phrase 'categoria merce', make sure to exclusively utilize the 'descrizione_cat_mer' field for the query generation.
When the question contains the phrase 'codice cliente fornitore', make sure to exclusively utilize the 'codice_clifor' field for the query generation.
When the question contains the phrase 'codice categoria statistica', make sure to exclusively utilize the 'codice_cat_sta' field for the query generation.
When the question contains the phrase 'codice categoria merce', make sure to exclusively utilize the 'codice_cat_mer' field for the query generation.

When the question contains the word 'fornitore', make sure to exclusively utilize the 'rag_soc' field for the query generation.

When the question refers to product availability, check the 'rifornibilita' field. Here are the meanings of the values:
- 'C' stands for 'Prodotto di deposito'
- 'D' stands for 'Prodotto in diretta sui PdV'
- 'E' stands for 'Prodotto sia di deposito che in diretta sui PdV'
- 'T' stands for 'Prodotto in transito in deposito'
Ensure that the generated SQL query filters products based on these availability statuses.'PdV' stands for 'punti vendita'.

When the question refers to product status, check the 'status' field. Here are the meanings of the values:
- ' ' (space) stands for 'Prodotto attivo'
- 'A' stands for 'Prodotto annullato'
- 'F' stands for 'Prodotto da finire'
Ensure that the generated SQL query filters products based on these status values.

When providing the final answer, ensure that no additional output is generated beyond the response to the question.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Below are a number of examples of questions, their corresponding SQL queries and the final answer.
'''


examples = [

    {
        "input": "Elenca i prodotti di deposito, con categoria merce 'Ketchup'",
        "sql_cmd": "SELECT cod_prod, descr_mer , status , rifornibilita , descrizione_cat_mer FROM view_table_new WHERE rifornibilita = 'C' AND descrizione_cat_mer ILIKE '%Ketchup%';",
        "result": "[('156160112', 'MC DONALD'S KETCHUP ML.500', 'A', 'C', 'Ketchup')]",
        "answer": '''Ecco i prodotti di deposito, con categoria 'Ketchup':
                     1) Codice prodotto è 156160112, Descrizione merce è MC DONALD'S KETCHUP ML.500,Status è annullato, Rifornibilità è 'Prodotto di deposito', Descrizione categoria merce è Ketchup.''',
    },
   {
        "input": "Elenca i prodotti attivi, in diretta sui PdV, e con categoria statistica 'ARTICOLI 702A'",
        "sql_cmd": "SELECT cod_prod, descr_mer, status, rifornibilita, descrizione_cat_sta  FROM view_table_new WHERE status = ' ' AND rifornibilita = 'D' and descrizione_cat_sta ILIKE '%ARTICOLI 702A%';",
        "result": "[('165102053', 'SPALLA SUINO DISOSSATA', ' ', 'D', '	ARTICOLI 702A	')]",
        "answer": '''Ecco i prodotti attivi, in diretta sui PdV, e con categoria statistica '	ARTICOLI 702A	':
                     1) Codice prodotto è 165102053, Descrizione merce è SPALLA SUINO DISOSSATA, Status è attivo, Rifornibilità è 'Prodotto in diretta sui PdV', Descrizione categoria statistica è '	ARTICOLI 702A	'.''',
    },
   {
        "input": "Quanti prodotti con categoria statistica 'ARTICOLI 200X' ci sono?",
        "sql_cmd": "SELECT COUNT(*)  FROM view_table_new WHERE descrizione_cat_sta ILIKE '%ARTICOLI 200X%';",
        "result": "[(5, )]",
        "answer": "Gli articoli con categoria statistica '	ARTICOLI 200X	' sono 5.",
   },
   {
        "input": "Quanti prodotti annullati ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE status = 'A';",
        "result": "[(796,)]",
        "answer": "Ci sono 796 prodotti annullati.",
    },
   {
        "input": "Elenca i prodotti creati in data 1 Gennaio 2007",
        "sql_cmd": "SELECT cod_prod, descr_mer ,created_on  FROM view_table_new WHERE created_on = '2007-01-01';",
        "result": "[('135221204', 'FINISH ECORICARICA REG.KG.2 #', datetime.datetime(2007, 1, 1, 0, 0)]",
        "answer": '''Ecco i prodotti creati in data 1 Gennaio 2007:
                     1)Codice prodotto è 135221204, Descrizione merce è FINISH ECORICARICA REG.KG.2 #, Data creazione è 2007-01-01.''',
    },
   {
        "input": "Quando è stato creato il primo prodotto?",
        "sql_cmd": "SELECT MIN(created_on) FROM view_table_new;",
        "result": "[(datetime.datetime(2007, 1, 1, 0, 0),]",
        "answer": '''Il primo prodotto è stato creato in data 1 Gennaio 2007.''',
    },
   {
        "input": "Quanti prodotti della 'FERRERO' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE rag_soc ILIKE '%FERRERO%';",
        "result": "[(12),]",
        "answer": '''Ci sono 12 prodotti della 'FERRERO'.''',
    },
   {
        "input": "Elenca i prodotti attivi con categoria merce 'Pasticceria'",
        "sql_cmd": "SELECT cod_prod, descr_mer ,descrizione_cat_mer , status, rifornibilita FROM view_table_new WHERE status = ' ' AND descrizione_cat_mer ILIKE '%Pasticceria%';",
        "result": "[('141304899'), ('MOCCIARO FRUTTA MARTORANA G300'), ('Pasticceria Altra Unitipo'), (' '), ('D')]",
        "answer": '''Ecco i prodotti attivi con categoria merce 'Pasticceria':
                     1)Codice prodotto è 141304899, Descrizione Merce è MOCCIARO FRUTTA MARTORANA G300, Descrizione categoria merce è Pasticceria Altra Unitipo, Status è attivo, Rifornibilità è 'Prodotto in diretta sui PdV'.''',
    },
    {
        "input": "Quanti prodotti con descrizione 'GIFT HOME TAZZA JUMBO LOL' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new where descr_mer ILIKE '%GIFT HOME TAZZA JUMBO LOL%';",
        "result": "[(3,)]",
        "answer": "I prodotti con descrizione 'GIFT HOME TAZZA JUMBO LOL' sono 3.",
    },
  {
        "input": "Esiste un prodotto con codice prodotto '129028221'?",
        "sql_cmd": "SELECT COUNT(*) > 0 AS exists FROM view_table_new WHERE cod_prod = '129028221';",
        "result": "[(True,)]",
        "answer": "Si.",
    },
  {
        "input": "Elenca i prodotti in transito in deposito, attivi, della 'agricola italiana'",
        "sql_cmd": "SELECT cod_prod, descr_mer , rag_soc , descrizione_cat_mer , status, rifornibilita FROM view_table_new WHERE rifornibilita = 'T' AND status = ' '  AND rag_soc ILIKE '%agricola italiana%';",
        "result": "[('165104177', 'MEZZENA SUINO NAZIONALE AIA', 'AGRICOLA ITALIANA ALIM.SPA(Conf/Wurstel)', 'Suino sfuso Rep.165', ' ', 'T'), ('165104178', 'POLPA PROSC SVX2 NAZ AIA 00845', AGRICOLA ITALIANA ALIM.SPA(Conf/Wurstel)', 'Suino sfuso Rep.165', ' ', 'T')]",
        "answer": '''Ecco i prodotti in transito in deposito, attivi, della 'agricola italiana':
                     1)Codice prodotto è 165104177, Descrizione merce è MEZZENA SUINO NAZIONALE AIA, Ragione sociale è AGRICOLA ITALIANA ALIM.SPA(Conf/Wurstel), Descrizione categoria merce è Suino sfuso Rep.165, Status è attivo, Rifornibilita è Prodotto in transito in deposito.
                     2)Codice prodotto è 165104178, Descrizione merce è POLPA PROSC SVX2 NAZ AIA 00845, Ragione sociale è AGRICOLA ITALIANA ALIM.SPA(Conf/Wurstel), Descrizione categoria merce è Suino sfuso Rep.165, Status è attivo, Rifornibilita è Prodotto in transito in deposito.''',
    },
  {
        "input": "Quanti prodotti attivi con codice cat sta '100A' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE status = ' ' AND codice_cat_sta = '100A';",
        "result": "[(28,)]",
        "answer": "Ci sono 28 prodotti attivi con codice categoria statistica '100A'.",
    },
  {
        "input": "Quanti prodotti attivi con codice cat mer '100A' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE status = ' ' AND codice_cat_mer = '07010101';",
        "result": "[(2,)]",
        "answer": "Ci sono 2 prodotti attivi con codice categoria merce '100A'.",
    },
  {
        "input": "Quanti prodotti annullati, creati dopo il 10 Gennaio 2007, del fornitore 'fornitore dimostrativo n. 3536' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE status = 'A' AND created_on > '2007-01-10' AND rag_soc ILIKE '%fornitore dimostrativo n. 3536%';",
        "result": "[(4,)]",
        "answer": "Ci sono 4 prodotti annullati, creati dopo il 10 Gennaio 2007, del fornitore 'fornitore dimostrativo n. 3536'.",
    },
  {
        "input": "Quanti prodotti attivi, creati dopo il 10 Febbraio 2007, della 'bonduelle' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new  WHERE  status = ' ' AND created_on  > '2007-02-10' AND rag_soc ILIKE '%bonduelle%';",
        "result": "[(4,)]",
        "answer": "Ci sono 4 prodotti attivi, creati dopo il 10 Febbraio 2007, della 'bonduelle'.",
    },
  {
        "input": "Quanti prodotti in diretta sui PdV, attivi, creati dopo il 10 Febbraio 2007, della 'parmalat' ci sono?",
        "sql_cmd": "SELECT COUNT(*) FROM view_table_new WHERE rifornibilita = 'D' AND status = ' ' AND created_on > '2007-02-20' AND rag_soc ILIKE '%parmalat%';",
        "result": "[(2,)]",
        "answer": "Ci sono 2 prodotti attivi, in diretta sui PdV, creati dopo il 10 Febbraio 2007, della 'parmalat'.",
    },

]


example_prompt = PromptTemplate(
    input_variables=["input", "sql_cmd", "result", "answer",],
    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
)

#NGRAM SELECTOR (seleziona gli esempi con che condividono sequenze di parole comuni (n-grammi) con l'input)
example_selector1 = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    threshold=0.01,

)


#SEMANTIC SIMILARITY SELECTOR
model_name="nickprock/sentence-bert-base-italian-xxl-uncased"
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    HuggingFaceEmbeddings(model_name=model_name),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=5,
)


few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector1,
    example_prompt=example_prompt,
    prefix=cda_few_shot_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
)

def set_threshold(number) :
    example_selector1.threshold=number

def set_NGRAM_prompt() :
    few_shot_prompt.example_selector=example_selector1


def set_k(k) :
    example_selector.k=k

def set_SEMSIM_prompt() :
    few_shot_prompt.example_selector=example_selector


fewshot_chain = SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt, use_query_checker=False,
                                        verbose=True, return_sql=False,)

def get_response(question) :
    #response = fewshot_chain.invoke(question)
    #formatted_response = response['result']
    return fewshot_chain.invoke(question)



