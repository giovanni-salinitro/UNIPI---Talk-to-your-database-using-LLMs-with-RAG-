from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import langchain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import NGramOverlapExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors.base import BaseExampleSelector

sqlcoder_mistral="/Users/macpro/.cache/lm-studio/models/MaziyarPanahi/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF/sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q8_0.gguf"
codellama_70b="/Users/macpro/.cache/lm-studio/models/TheBloke/CodeLlama-70B-Instruct-GGUF/codellama-70b-instruct.Q5_K_M.gguf"

llm = LlamaCpp(
        model_path=sqlcoder_mistral,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=10000,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        repeat_penalty=1.3,
        temperature=0.0,
        max_tokens = 512
    )

langchain.verbose = False 
pg_uri = f"clickhouse+http://tirocinio:#Tirocinio!2024@192.168.89.8:8123/chpv"
db = SQLDatabase.from_uri( pg_uri,
                           include_tables=['view_pv_tabellone'],
                           sample_rows_in_table_info=0
                         )


cda_fewshot_prompt='''
You are a ClickHouse expert. Given an input question, first create a syntactically correct ClickHouse query to run, then look at the results of the query and return the answer in Italian language to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per ClickHouse. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
The table on which you will run SQL queries is a table that contains the products sold by various outlets, with the various categories, departments, quantity sold, sales value, etc.
The various elements (stores, categories, products, departments) have both a code and a description. The user will often refer to the description, in some cases he may refer to the code; in both cases their value will be enclosed in single quotes (') or double quotes ('').

FIRST YOU HAVE TO CREATE THE SQL QUERY, NEVER PROVIDE THE ANSWER BEFORE GENERATE THE SQL QUERY AND THEN GIVE THE FINAL ANSWER BASED ON THE SQL RESULT.

Preserve strings, numbers, and spaces in quotes in the question as they are in the generated SQL query; please pay attention to this.

Write alias (AS) without enclose it in quotes in the generated SQL query. Please pay attention to this.

For dates always use the "=" operator.

When the question involves searching or calculating a result based on a specific field such as sale point, department, or brand, ensure to include a GROUP BY clause to group the results by that field. This is necessary to aggregate the data correctly and obtain the desired outcome.

When the question refers to the quantity sold, be careful to calculate it with the formula "SUM(qta_offerta + qta_non_offerta)" in the generated SQL query ("qta_offerta" and "qta_non_offerta" are two fields of the table).
When the question refers to the sold value, be careful to calculate it with the formula "SUM(val_off + val_non_off)" in the generated SQL query ("val_off" and "val_non_off" are two fields in the table).
When the question refers to cost of goods sold, be careful to calculate it with the formula "SUM(costo_nettissimo_off + costo_nettissimo_no_off)" in the generated SQL query ("costo_nettissimo_off" and "costo_nettissimo_no_off" are two fields in the table).
When the question contains the word "margine", be careful to calculate its numeric value with the formula "SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off)", and its value in percentage (%) with the formula "SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off)" in the generated SQL query ("val_off" , "val_non_off", "costo_nettissimo_off" and "costo_nettissimo_no_off" are fields on the table).
When the question contains the phrase 'incidenza delle offerte', be careful to calculate it with the formula "SUM(val_off) / SUM(val_off + val_non_off)" in the generated SQL query ("val_off" and "val_non_off" are two fields in the table).

BE EXTREMELY CAREFUL TO USE ONE OF THE GIVEN FORMULAS TO CALCULATE THE VARIOUS VALUES WITHOUT INVENTING ONE.

When you use the "rag_soc" field ALWAYS use the LIKE operator in the generated SQL query. 
When you use the "descr_cat" field ALWAYS use the LIKE operator in the generated SQL query.

NEVER use 'data_doc' field and 'data_format_date' field with LIKE operator.

When the question refers to a sale point sign, use the 'descr_cat' field and the LIKE operator with the '%' symbol to match similar patterns in the generated SQL query.
When the question refers to the department of a sale point, use the 'descr_liv1' field and the LIKE operator with the '%' symbol to match similar patterns in the generated SQL query.
When the question refers to a sale point enclosed in single quotes (' '), use the 'rag_soc' field and the LIKE operator with the '%' symbol to match similar patterns in the generated SQL query to filter by name point of sale.

When the question refers to a date , use the data_doc field in YYYYMMDD format (data_doc= YYYYMMDD). For example data_doc = 20210215. Put the date in the generated query ONLY if specified. IGNORE data_format_date field.
When you have to do a COUNT of some product, be careful to use the DISTINCT operator because in each store there are multiple products with the same product code (cod_prod); this is because the table takes into account the history of products sold every day. Then use COUNT(DISTINCT cod_prod) for example.
When the question contains the word 'insegna', make sure to exclusively utilize the 'descr_cat' field for the query generation with the LIKE operator.


Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Below are a number of examples of questions, their corresponding SQL queries and the final answer.
'''

examples = [

   {
        "input": "Quanti prodotti ha venduto il punto vendita 'SUPERSTORE 01' nel mese di Gennaio 2021?",
        "sql_cmd": "SELECT COUNT(DISTINCT cod_prod) AS quantita_prodotti_venduti FROM view_pv_tabellone WHERE rag_soc LIKE '%SUPERSTORE 01%' AND data_doc >= '20210101' AND data_doc <= '20210131';",
        "result": "[(15171, )]",
        "answer": '''Il punto vendita 'SUPERSTORE 01' ha venduto nel mese di Gennaio 2021 15.171 prodotti.'''
    },
   {
        "input": "Quanti prodotti ha venduto il punto vendita 'SUPERSTORE 02'?",
        "sql_cmd": "SELECT COUNT(DISTINCT cod_prod) AS prodotti_venduti FROM view_pv_tabellone WHERE rag_soc LIKE '%SUPERSTORE 02%';",
        "result": "[(23227, )]",
        "answer": '''Il punto vendita 'superstore 02' ha venduto 23.227 prodotti.'''
    },
    {
        "input": "Quale punto vendita ha la maggior quantità venduta per punto vendita?",
        "sql_cmd": "SELECT rag_soc, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone GROUP BY rag_soc ORDER BY quantita_venduta DESC LIMIT 1;",
        "result": "[('SUPERSTORE 01', '20149100.846999988')]",
        "answer": '''Il punto vendita con la maggior quantità venduta è 'SUPERSTORE 01' e la quantità venduta è 20149100.846999988.'''
    },
    {
        "input": "Quale è la quantità venduta del punto vendita 'SUPERMARKET 01' in data 31 Dicembre 2020?",
        "sql_cmd": "SELECT SUM(qta_offerta+qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE rag_soc LIKE '%SUPERMARKET 01%' AND data_doc = '20201231';",
        "result": "[('7627.214', )]",
        "answer": '''La quantità venduta del punto vendita 'SUPERMARKET 01' in data 31 Dicembre 2020 è 7627.214.'''
    },
    {
        "input": "Elenca i primi 2 punti vendita con la minor quantità venduta per punto vendita",
        "sql_cmd": "SELECT rag_soc AS punto_vendita, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone GROUP BY rag_soc ORDER BY quantita_venduta ASC LIMIT 2;",
        "result": "[('SUPERMARKET 07', '1628323.6799999978'),('SUPERMARKET 13', '1876560.0540000023')]",
        "answer": '''Ecco i primi 2 punti vendita con la minor quantità venduta per punto vendita:
                     1)Punto vendita:SUPERMARKET 07 , Quantità venduta:1628323.6799999978.
                     2)Punto vendita:SUPERMARKET 13 , Quantità venduta:1876560.0540000023.'''
    },
    {
        "input": "Elenca i primi 2 punti vendita con il maggior valore venduto per punto vendita nel mese di Gennaio 2021",
        "sql_cmd": "SELECT rag_soc , SUM(val_off + val_non_off) as valore_venduto FROM view_pv_tabellone WHERE data_doc >= 20210101 AND data_doc <= 20210131 GROUP BY rag_soc ORDER BY valore_venduto DESC LIMIT 2;",
        "result": "[('SUPERSTORE 01', '1682122.2177758007'),('SUPERSTORE 04', '1617361.7323211778')]",
        "answer": '''Ecco i primi 2 punti vendita con il maggior valore venduto per punto vendita:
                     1)Punto vendita:SUPERSTORE 01 , Valore venduto:1682122.2177758007.
                     2)Punto vendita:SUPERSTORE 04 , Valore venduto:1617361.7323211778.'''
    },
    {
        "input": "Quale è il valore venduto del punto vendita 'IPERSTORE 01' nel mese di Gennaio 2020?",
        "sql_cmd": "SELECT SUM(val_off + val_non_off) as valore_venduto FROM view_pv_tabellone WHERE rag_soc LIKE '%IPERSTORE 01%' AND data_doc >= 20200101 AND data_doc <= 20200131;",
        "result": "[('1210787.3670404728', )]",
        "answer": '''Il valore venduto del punto vendita 'IPERSTORE 01' nel mese di Gennaio 2020 è 1210787.3670404728.'''
    },
    {
        "input": "Quale è il punto vendita con il maggior valore venduto per punto vendita in data 30 Gennaio 2021?",
        "sql_cmd": "SELECT rag_soc, SUM(val_off + val_non_off) as valore_venduto FROM view_pv_tabellone WHERE data_doc = 20210130 GROUP BY rag_soc ORDER BY valore_venduto DESC LIMIT 1;",
        "result": "[('IPERSTORE 03', '42854.56832975754')]",
        "answer": '''Il punto vendita con il maggior valore venduto in data 30 Gennaio 2021 è 'IPERSTORE 03' con un valore venduto di 42854.56832975754.'''
    },
    {
        "input": "Quale punto vendita ha il maggior costo del venduto per punto vendita nel 2021?",
        "sql_cmd": "SELECT rag_soc , sum(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE data_doc >= 20210101 AND data_doc <= 20211231 GROUP BY rag_soc ORDER BY costo_venduto DESC LIMIT 1;",
        "result": "[('SUPERSTORE 01', '14267935.810414173')]",
        "answer": '''Il punto vendita con il maggior costo del venduto nel 2021 è il punto vendita 'SUPERSTORE 01' e il costo del venduto è di 4267935.810414173.'''
    },
    {
        "input": "Quale punto vendita ha il minor costo del venduto per punto vendita nel 2020?",
        "sql_cmd": "SELECT rag_soc , sum(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20201231 GROUP BY rag_soc ORDER BY costo_venduto ASC LIMIT 1;",
        "result": "[('SUPERSTORE 01', '14267935.810414173')]",
        "answer": '''Il punto vendita con il minor costo del venduto nel 2021 è il punto vendita 'SUPERSTORE 01' e il costo del venduto è di 4267935.810414173.'''
    },
    {
        "input": "Quale insegna ha il minor costo del venduto per insegna nel 2020?",
        "sql_cmd": "SELECT descr_cat , SUM(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20201231 GROUP BY descr_cat ORDER BY costo_venduto ASC;",
        "result": "[('SUPERMARKET', '30400033.030205924')]",
        "answer": '''L'insegna con il minor costo del venduto per insegna nel 2020 è 'SUPERMARKET' e il costo del venduto è di 30400033.030205924.'''
    },
    {
        "input": "Elenca il costo del venduto per insegna nel mese di Gennaio 2020",
        "sql_cmd": "SELECT descr_cat AS insegna , SUM(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20200131 GROUP BY descr_cat ORDER BY costo_venduto DESC;",
        "result": "[('SUPERSTORE', '122716576.59607387'), ('IPERSTORE', '37516208.06436576'), ('SUPERMARKET', '30400033.030205924')]",
        "answer": '''Ecco i costi del venduto per insegna nel 2020:
                     1)Insegna:SUPERSTORE , Costo del venduto:122716576.59607387.
                     2)Insegna:IPERSTORE , Costo del venduto:37516208.06436576.
                     3)Insegna:SUPERMARKET , Costo del venduto:30400033.030205924.'''
    },
    {
        "input": "Elenca la quantità venduta per insegna nel mese di Gennaio 2020",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20200131 GROUP BY descr_cat ORDER BY quantita_venduta DESC;",
        "result": "[('SUPERSTORE', '122716576.59607387'), ('IPERSTORE', '37516208.06436576'), ('SUPERMARKET', '30400033.030205924')]",
        "answer": '''Ecco la quantità venduta per insegna:
                     1)Insegna:SUPERSTORE , Quantità venduta:122716576.59607387.
                     2)Insegna:IPERSTORE , Quantità venduta:37516208.06436576.
                     3)Insegna:SUPERMARKET , Quantità venduta:30400033.030205924.'''
    },
    {
        "input": "Quale insegna ha la maggior quantità venduta nel mese di Gennaio 2020?",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20200131 GROUP BY descr_cat ORDER BY quantita_venduta DESC;",
        "result": "[('SUPERSTORE', '122716576.59607387'), ('IPERSTORE', '37516208.06436576'), ('SUPERMARKET', '30400033.030205924')]",
        "answer": '''L'insegna 'SUPERSTORE' ha la maggior quantità venduta nel mese di Gennaio 2020 con una quantità venduta di 122716576.59607387.'''
    },
    {
        "input": "Elenca il valore venduto per insegna nel mese di Gennaio 2020",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE data_doc >= 20200101 AND data_doc <= 20200131 GROUP BY descr_cat ORDER BY valore_venduto DESC;",
        "result": "[('SUPERSTORE', '122716576.59607387'), ('IPERSTORE', '37516208.06436576'), ('SUPERMARKET', '30400033.030205924')]",
        "answer": '''Ecco il valore venduto per insegna nel mese di Gennaio 2020:
                     1)Insegna:SUPERSTORE , Valore venduto:122716576.59607387.
                     2)Insegna:IPERSTORE , Valore venduto:37516208.06436576.
                     3)Insegna:SUPERMARKET , Valore venduto:30400033.030205924.'''
    },
    {
        "input": "Quale insegna ha il maggior costo del venduto nel mese di Marzo 2020?",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE data_doc >= 20200301 AND data_doc <= 20200331 GROUP BY descr_cat ORDER BY costo_venduto DESC;",
        "result": "[('SUPERSTORE', '122716576.59607387'), ('IPERSTORE', '37516208.06436576'), ('SUPERMARKET', '30400033.030205924')]",
        "answer": '''L'insegna 'SUPERSTORE' ha il maggior costo del venduto nel mese di Marzo 2020 con una quantità venduta di 122716576.59607387.'''
    },
    {
        "input": "Quale insegna ha il minor valore venduto in data 11 Aprile 2021?",
        "sql_cmd": "SELECT descr_cat AS insegna , SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE data_doc = 20210411 GROUP BY descr_cat ORDER BY valore_venduto ASC LIMIT 1;",
        "result": "[('SUPERMARKET', '16427.257306218355')]",
        "answer": '''L'insegna 'SUPERMARKET' ha il minor valore venduto in data 11 Aprile 2021 con una valore venduto di 16427.257306218355.'''
    },
    {
        "input": "Quale è il valore venduto dell'insegna 'SUPERMARKET' in data 11 Dicembre 2021?",
        "sql_cmd": "SELECT SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE descr_cat LIKE '%SUPERMARKET%' AND data_doc = 20211211;",
        "result": "[('169254.83145215438')]",
        "answer": '''Il valore venduto dell'insegna 'SUPERMARKET' in data 11 Dicembre 2021 è di 169254.83145215438.'''
    },
    {
        "input": "Quale è l'insegna con la minor quantità venduta in data 2 Maggio 2020?",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE data_doc = 20200502 GROUP BY descr_cat ORDER BY quantita_venduta ASC LIMIT 1;",
        "result": "[('IPERSTORE', '82756.12499999994')]",
        "answer": '''L'insegna con la minor quantità venduta in data 2 Maggio 2020 è 'IPERSTORE' e la quantità venduta è di 82756.12499999994.'''
    },
    {
        "input": "Elenca la quantità venduta per reparto dei primi 3 reparti del punto vendita 'SUPERSTORE 01' in data 12 Novembre 2020",
        "sql_cmd": "SELECT descr_liv1 AS reparto, SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE data_doc = 20201112 AND rag_soc LIKE '%SUPERSTORE 01%' GROUP BY descr_liv1 ORDER BY quantita_venduta DESC LIMIT 3;",
        "result": "[('Grocery', '6485.0'), ('Fresh', '4775.590000000006'), ('Beverages', '2836.0')]",
        "answer": '''Ecco  la quantità venduta per reparto dei primi 3 reparti del punto vendita 'SUPERSTORE 01' in data 12 Novembre 2020:
                     1)Reparto:Grocery , Quantità venduta:6485.0.
                     2)Reparto:Fresh , Quantità venduta:4775.59.
                     3)Reparto:Beverages , Quantità venduta:2836.0.'''
    },
    {
        "input": "Quale è il reparto con la minor quantità venduta del punto vendita 'IPERSTORE 02' nel mese di Giugno 2021?",
        "sql_cmd": "SELECT descr_liv1 AS reparto , SUM(qta_offerta + qta_non_offerta) AS quantita_venduta FROM view_pv_tabellone WHERE rag_soc LIKE '%IPERSTORE 02%' AND data_doc >= 20210601 AND data_doc <= 20210631 GROUP BY descr_liv1 ORDER BY quantita_venduta ASC LIMIT 1;",
        "result": "[('Others', '166.0')]",
        "answer": '''Il reparto con la minor quantità venduta del punto vendita 'IPERSTORE 02' nel mese di Giugno 2021 è 'Others' e la quantità venduta è 166.'''
    },
    {
        "input": "Quale è il valore venduto del reparto 'Grocery' del punto vendita 'SUPERSTORE 02' in data 13 Gennaio 2021?",
        "sql_cmd": "SELECT SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE rag_soc LIKE '%SUPERSTORE 02%' AND data_doc = 20210113 AND descr_liv1 LIKE '%Grocery%';",
        "result": "[('10985.448747836863')]",
        "answer": '''Il valore venduto del reparto 'Grocery' del punto vendita 'SUPERSTORE 02' in data 13 Gennaio 2021.'''
    },
    {
        "input": "Quale è il reparto con il maggior valore venduto per reparto del punto vendita 'IPERSTORE 02' nel mese di Giugno 2021?",
        "sql_cmd": "SELECT descr_liv1 AS reparto , SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE rag_soc LIKE '%IPERSTORE 02%' AND data_doc >= 20210601 AND data_doc <= 20210631 GROUP BY descr_liv1 ORDER BY valore_venduto DESC LIMIT 1;",
        "result": "[('Grocery', '288196.1625415588')]",
        "answer": '''Il reparto con il maggior valore venduto per reparto del punto vendita 'IPERSTORE 02' nel mese di Giugno 2021 è 'Grocery' e il valore venduto è di '288196.1625415588'.'''
    },
    {
        "input": "Elenca il valore venduto per reparto dei primi 3 reparti del punto vendita 'SUPERMARKET 01' in data 4 Gennaio 2020",
        "sql_cmd": "SELECT descr_liv1 AS reparto , SUM(val_off + val_non_off) AS valore_venduto FROM view_pv_tabellone WHERE rag_soc LIKE '%SUPERMARKET%01%' AND data_doc = 20200104 GROUP BY descr_liv1 ORDER BY valore_venduto DESC LIMIT 3;",
        "result": " [('Fresh', '12097.424661813593'), ('Grocery', '10945.976865484232'), ('Fruit and Vegetables', '4246.007757133577')]",
        "answer": '''Ecco il valore venduto per reparto dei primi 3 reparti del punto vendita 'SUPERSTORE 01' in data 4 Gennaio 2020:
                     1)Reparto:Fresh , Valore venduto:12097.424661813593.
                     2)Reparto:Grocery , Valore venduto:10945.976865484232.
                     3)Reparto:Fruit and Vegetables , Valore venduto:4246.007757133577.'''
    },
    {
        "input": "Quale è il costo del venduto del reparto 'Fresh' del punto vendita 'IPERSTORE 03' in data 4 Luglio 2021?",
        "sql_cmd": "SELECT SUM(costo_nettissimo_off+costo_nettissimo_no_off) AS costo_venduto FROM view_pv_tabellone WHERE rag_soc LIKE '%IPERSTORE 03%' AND data_doc = 20210704 AND descr_liv1 LIKE '%Fresh%';",
        "result": "[('5692.604330378268')]",
        "answer": '''Il costo del venduto del reparto 'Fresh' del punto vendita 'IPERSTORE 03' in data 4 Luglio 2021 è 5692.604330378268.'''
    },
    {
        "input": "Elenca il margine per punto vendita dei primi 3 punti vendita nel mese di Dicembre 2021",
        "sql_cmd": "SELECT rag_soc , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc >= 20211201 AND data_doc <= 20211231 GROUP BY rag_soc ORDER BY margine DESC LIMIT 3;",
        "result": "[('SUPERSTORE 01', '797047.4521092057', '0.31549309135079445'), ('SUPERSTORE 04' '622730.7819912424', '0.32284551927512095'), ('IPERSTORE 01', '537157.895063902', '0.2825562576572131')]",
        "answer": '''Ecco il margine per punto vendita dei primi 3 punti vendita nel mese di Dicembre 2021:
                     1)Punto vendita:SUPERSTORE 01, Margine:797047.4521092057, Margine in percentuale:31,54%.
                     2)Punto vendita:SUPERSTORE 04, Margine:622730.7819912424, Margine in percentuale:32,28%.
                     3)Punto vendita:IPERSTORE 01, Margine:537157.895063902, Margine in percentuale:28,25%.'''
    },
    {
        "input": "Elenca il margine per punto vendita , dei primi 2 punti vendita , in data 7 Settembre 2020",
        "sql_cmd": "SELECT rag_soc , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc = 20200907 GROUP BY rag_soc ORDER BY margine DESC LIMIT 2;",
        "result": "[('SUPERSTORE 01', '15400.038950255586', '0.3381113508077541'), ('IPERSTORE 03' '12468.405150869175', '0.3409703844561733')]",
        "answer": '''Ecco il margine per punto vendita , dei primi 2 punti vendita , in data 7 Settembre 2020:
                     1)Punto vendita:SUPERSTORE 01, Margine:15400.038950255586, Margine in percentuale:33,81%.
                     2)Punto vendita:IPERSTORE 03, Margine:12468.405150869175, Margine in percentuale:34,09%.'''
    },
    {
        "input": "Quale è il punto vendita con il maggior margine per punto vendita in data 7 Settembre 2021?",
        "sql_cmd": "SELECT rag_soc , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc = 20210907 GROUP BY rag_soc ORDER BY margine DESC LIMIT 1;",
        "result": "[('SUPERSTORE 01', '19706.46311606039', '0.35928738793859044')]",
        "answer": '''Il punto vendita con il maggior margine per punto vendita in data 7 Settembre 2021 è il punto vendita 'SUPERSTORE 01', con margine 19706.46311606039 e margine in percentuale 35,92%.'''
    },
    {
        "input": "Quale è il margine del punto vendita 'IPERSTORE 03' in data 7 Settembre 2021?",
        "sql_cmd": "SELECT SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc = 20210907 AND rag_soc LIKE '%IPERSTORE 03%';",
        "result": "[('11537.523151947957', '0.3507536478832071')]",
        "answer": '''Il margine del punto vendita 'IPERSTORE 03' in data 7 Settembre 2021 è 11537.523151947957 e il margine in percentuale è 35,07%'''
    },
    {
        "input": "Elenca il margine per insegna di tutte le insegne in data 30 Gennaio 2020",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc = 20200130 GROUP BY insegna ORDER BY margine DESC;",
        "result": "[('SUPERSTORE', '112462.12621075862', '0.28479469300958865'), ('IPERSTORE' '33738.902300438946', '0.2929215850328978'), ('SUPERMARKET', '32943.536555229024', '0.2910049926241966')]",
        "answer": '''Ecco il margine per punto vendita dei primi 3 punti vendita in data 30 Gennaio 2020:
                     1)Insegna:SUPERSTORE, Margine:112462.12621075862, Margine in percentuale:28,47%.
                     2)Insegna:IPERSTORE, Margine:33738.902300438946, Margine in percentuale:29,29%.
                     3)Insegna:SUPERMARKET, Margine:32943.536555229024, Margine in percentuale:29,10%.'''
    },
    {
        "input": "Quale insegna ha il minor margine per insegna in data 30 Maggio 2020?",
        "sql_cmd": "SELECT descr_cat AS insegna, SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc = 20200530 GROUP BY insegna ORDER BY margine ASC LIMIT 1;",
        "result": "[('11537.523151947957', '0.3507536478832071')]",
        "answer": '''Il margine del punto vendita 'IPERSTORE 03' in data 7 Settembre 2021 è 11537.523151947957 e il margine in percentuale è 35,07%'''
    },
    {
        "input": "Elenca il margine per reparto dei primi 3 reparti del punto vendita 'IPERSTORE 03' nel mese di Luglio 2020",
        "sql_cmd": "SELECT descr_liv1 AS reparto , SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) AS margine, SUM(val_off + val_non_off - costo_nettissimo_off - costo_nettissimo_no_off) / SUM(val_off + val_non_off) AS margine_percentuale FROM view_pv_tabellone WHERE data_doc >= 20200701 AND data_doc <= 20200731 AND rag_soc LIKE '%IPERSTORE 03%' GROUP BY descr_liv1 ORDER BY margine DESC LIMIT 3;",
        "result": "[('Fresh', '73068.36458870037', '0.33733753651831366'), ('Grocery' '69796.45618823804', '0.30222769568799535'), ('Fish', '38032.61325545446', '0.6374601009621605')]",
        "answer": '''Ecco il margine per reparto , dei primi 3 reparti , del punto vendita 'IPERSTORE 03' nel mese di Luglio 2020:
                     1)Reparto:Fresh, Margine:73068.36458870037, Margine in percentuale:33,73%.
                     2)Reparto:Grocery, Margine:69796.45618823804, Margine in percentuale:30,22%.
                     3)Reparto:Fish, Margine:38032.61325545446, Margine in percentuale:63,74%.'''
    },
    {
        "input": "Elenca l'incidenza delle offerte per punto vendita dei primi 3 punti vendita in data 22 Marzo 2021",
        "sql_cmd": "SELECT rag_soc AS punto_vendita , SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20210322 GROUP BY rag_soc ORDER BY incidenza_offerte DESC LIMIT 3;",
        "result": "[('SUPERMARKET 09', '0.42819371496018116'), ('SUPERMARKET 10' '0.4056609388324042'), ('SUPERMARKET 07', '0.3858596708109372')]",
        "answer": '''Ecco l'incidenza delle offerte per punto vendita dei primi 3 punti vendita in data 22 Marzo 2021:
                     1)Punto vendita:SUPERMARKET 09, Incidenza offerte:42,81%.
                     2)Punto vendita:SUPERMARKET 10, Incidenza offerte:40,56%.
                     3)Punto vendita:SUPERMARKET 07, Incidenza offerte:38,58%.'''
    },
    {
        "input": "Quale è l'incidenza delle offerte del punto vendita 'SUPERSTORE 11' in data 29 Aprile 2020?",
        "sql_cmd": "SELECT SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20200429 AND rag_soc LIKE '%SUPERSTORE 11%';",
        "result": "[('0.3491785193853042')]",
        "answer": '''L'incidenza delle offerte del punto vendita 'SUPERSTORE 11' in data 29 Aprile 2020 è del 34,91%'''
    },
    {
        "input": "Quale punto vendita ha la maggior incidenza delle offerte in data 3 Febbraio 2021",
        "sql_cmd": "SELECT rag_soc AS punto_vendita , SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20210203 GROUP BY rag_soc ORDER BY incidenza_offerte DESC LIMIT 1;",
        "result": "[('SUPERMARKET 09', '0.37855651084082753')]",
        "answer": '''Il punto vendita con la maggior incidenza delle offerte in data 3 Febbraio 2021 è il punto vendita 'SUPERMARKET 09' e l'incidenza delle offerte è 37,85%.'''
    },
    {
        "input": "Quale è l'incidenza delle offerte dell'insegna 'SUPERSTORE' in data 17 Settembre 2020?",
        "sql_cmd": "SELECT SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20200917 AND descr_cat LIKE '%SUPERSTORE%';",
        "result": "[(0.19891156626795162')]",
        "answer": '''L'incidenza delle offerte dell'insegna 'SUPERSTORE' in data 17 Settembre 2020 è del 19,89%'''
    },
    {
        "input": "Elenca l'incidenza delle offerte per insegna di tutte le insegne in data 6 Dicembre 2020",
        "sql_cmd": "SELECT descr_cat , SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20201206 GROUP BY descr_cat ORDER BY incidenza_offerte DESC;",
        "result": "[('SUPERSTORE', '0.33559998948799796'), ('IPERSTORE', '0.2827295970557721'), ('SUPERMARKET', '0.2761885210734631')]",
        "answer": '''Ecco l'incidenza delle offerte per insegna di tutte le insegne in data 6 Dicembre 2020:
                     1)Insegna:SUPERSTORE , Incidenza offerte:33,55%.
                     2)Insegna:IPERSTORE , Incidenza offerte:28,27%.
                     3)Insegna:SUPERMARKET , Incidenza offerte:27,61%.'''
    },
    {
        "input": "Quale insegna ha la maggior incidenza delle offerte in data 14 Giugno 2021",
        "sql_cmd": "SELECT descr_cat AS insegna , SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20210614 GROUP BY descr_cat ORDER BY incidenza_offerte DESC LIMIT 1;",
        "result": "[('SUPERMARKET', '0.3097916534984438')]",
        "answer": '''L'insegna con la maggior incidenza delle offerte in data 14 Giugno 2021 è l'insegna 'SUPERMARKET' e l'incidenza delle offerte è 30,97%.'''
    },
    {
        "input": "Elenca  per insegna in data 6 Dicembre 2020",
        "sql_cmd": "SELECT descr_cat , SUM(val_off) / SUM(val_off + val_non_off) AS incidenza_offerte FROM view_pv_tabellone WHERE data_doc = 20201206 GROUP BY descr_cat ORDER BY incidenza_offerte DESC;",
        "result": "[('SUPERSTORE', '0.33559998948799796'), ('IPERSTORE', '0.2827295970557721'), ('SUPERMARKET', '0.2761885210734631')]",
        "answer": '''Ecco il valore venduto per insegna nel mese di Gennaio 2020:
                     1)Insegna:SUPERSTORE , Incidenza offerte:33,55%.
                     2)Insegna:IPERSTORE , Incidenza offerte:28,27%.
                     3)Insegna:SUPERMARKET , Incidenza offerte:27,61%.'''
    },

]


example_prompt = PromptTemplate(
    input_variables=["input", "sql_cmd", "result", "answer",],
    template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
)

#NGRAM EXAMPLE SELECTOR
example_selector1 = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    threshold=0.07,
)

#SEMANTIC SIMILARITY EXAMPLE SELECTOR
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

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        input_text = input_variables["input"].lower()

        # Prima ricerca: parole chiave specifiche
        specific_keywords = ["reparto", "reparti", "punto vendita", "punti vendita", "insegne","insegna"]
        selected_examples = []
        for example in self.examples:
            example_text = example["input"].lower()
            for keyword in specific_keywords:
                if keyword in input_text:
                    if keyword in example_text:
                        selected_examples.append(example)
                        break

        # Seconda ricerca: parole chiave generali
        general_keywords = ["margine", "valore venduto", "incidenza delle offerte", "quantità venduta", "costo del venduto"]
        second_pass_selected_examples = []
        for example in selected_examples:
            example_text = example["input"].lower()
            for keyword in general_keywords:
                if keyword in input_text:
                    if keyword in example_text:
                        second_pass_selected_examples.append(example)
                        break
        
        # Rimuovere i duplicati dalla lista di esempi selezionati
        unique_examples = {tuple(example.values()): example for example in second_pass_selected_examples}.values()

        if len(unique_examples) !=0 :
            return list(unique_examples)
        else:
            return selected_examples

example_selector2 = CustomExampleSelector(examples)

CUSTOM_SUFFIX ='''

Only use the following tables:
CREATE TABLE view_pv_tabellone (
    cod_cli_for Int32,
    rag_soc String,
    cat_cli String,
    descr_cat String,    
    data_doc Int32,
    data_format_date DATE,
    cod_prod String,
    descr_prod String,
    liv1 String,
    descr_liv1 String,
    tipologia_prod String,
    qta_offerta Float64,
    qta_non_offerta Float64,
    val_off Float64,
    val_non_off Float64,
    costo_nettissimo_off Float64,
    costo_nettissimo_no_off Float64
) ENGINE = MergeTree()
 PARTITION BY anno
 ORDER BY (data_format_date, cod_cli_for, cod_prod)
 PRIMARY KEY (data_format_date, cod_cli_for, cod_prod)

Question: {input}

'''

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector1,
    example_prompt=example_prompt,
    prefix=cda_fewshot_prompt,
    suffix=CUSTOM_SUFFIX,
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

def set_CUSTOMSELECTOR_prompt() :
    few_shot_prompt.example_selector=example_selector2


fewshot_chain = SQLDatabaseChain.from_llm(llm, db, prompt=few_shot_prompt, use_query_checker=False,
                                        verbose=True, return_sql=False,)

def get_response(question) :
    #response = fewshot_chain.invoke(question)
    #formatted_response = response['result']
    return fewshot_chain.invoke(question)
