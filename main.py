import streamlit as st
import pandas as pd
from langchain_groq.chat_models import ChatGroq
from io import StringIO

st.set_page_config(page_title="AI-Powered Attribute-Level SQL Translator", layout="wide")

st.title("🤖 Attribute-Level Business Requirement → SQL Translator (with CSV Export)")

st.markdown("""
Upload or enter your requirements and get back a CSV with **SQL Output** generated.  
Supports both simple and complex logic (aggregations, CASE, window functions, joins, etc.).
""")

# ---- API Key ----
api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.6
)

# ---- INPUT ----
uploaded_file = st.file_uploader("Upload a CSV file with requirements", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Or enter requirements manually below")
    df = pd.DataFrame(columns=["TargetObject", "TargetObjectAttribute", "CalculationLogic", "SQLContext"])

# Editable grid
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ---- LLM TRANSLATION ----
def generate_sql_with_llm(attr, logic, target_object=None, context=None):
    """
    Uses Groq LLM to generate SQL expressions for attributes.
    """
    prompt = f"""
    You are an expert SQL generator for Databricks.
    Convert the following requirement into a valid SQL expression (not a full query).
    
    Target Object: {target_object}
    Attribute: {attr}
    Requirement (English): {logic}
    Context (if any): {context}
    
    Rules:
    - Return ONLY the SQL expression for the attribute, not a full SELECT query.
    - Alias the expression as the attribute name: AS {attr}.
    - If CASE WHEN or joins are required, write them inline in the expression.
    - Keep SQL syntax strictly compatible with Databricks SQL.
    """

    try:
        response = llm.invoke(prompt)
        sql_expr = response.content.strip()
        return sql_expr
    except Exception as e:
        return f"-- ERROR generating SQL: {str(e)}"

# ---- PROCESS ----
if st.button("Generate SQL and Export CSV"):
    if edited_df.empty:
        st.warning("Please provide some input data.")
    else:
        # Generate SQL expressions
        edited_df["SQL Output"] = edited_df.apply(
            lambda row: generate_sql_with_llm(
                attr=row["TargetObjectAttribute"],
                logic=row["CalculationLogic"],
                target_object=row.get("TargetObject", ""),
                context=row.get("SQLContext", "")
            ), axis=1
        )

        st.subheader("Generated SQL Outputs")
        st.dataframe(edited_df, use_container_width=True)

        # Convert to CSV
        csv_buffer = StringIO()
        edited_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="sql_output.csv",
            mime="text/csv"
        )
