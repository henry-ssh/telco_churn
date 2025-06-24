import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

# 🚩 Carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv("telco_customer_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# 🔸 Cálculos dos KPIs
total_clientes = len(df)
clientes_ativos = df[df['Churn'] == 'No'].shape[0]
clientes_cancelados = df[df['Churn'] == 'Yes'].shape[0]

taxa_churn = (clientes_cancelados / total_clientes) * 100

arpu = df['MonthlyCharges'].mean()

receita_perdida = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()

tenure_medio = df['tenure'].mean()

ticket_medio = df['TotalCharges'].mean()

st.title("📊 Dashboard - Análise de Churn")

st.subheader("🔎 KPIs Principais")

col1, col2, col3 = st.columns(3)
col1.metric("Taxa de Churn", f"{taxa_churn:.2f}%")
col2.metric("Clientes Ativos", f"{clientes_ativos}")
col3.metric("Clientes Cancelados", f"{clientes_cancelados}")

col4, col5, col6 = st.columns(3)
col4.metric("Receita Média Mensal", f"R$ {arpu:.2f}")
col5.metric("Receita Perdida (Churn)", f"R$ {receita_perdida:.2f}")
col6.metric("Tempo Médio (Tenure)", f"{tenure_medio:.0f} meses")

st.divider()

# 📈 Sidebar
st.sidebar.header("Configurações de Visualização")
show_distributions = st.sidebar.checkbox("Distribuição de Dados Categóricos", True)
show_numeric = st.sidebar.checkbox("Análise Variáveis Numéricas", True)
show_churn = st.sidebar.checkbox("Taxa de Cancelamento (Churn)", True)
show_bivariadas = st.sidebar.checkbox("Análise Bivariada (Churn)", True)

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido por Senhor")


# =======================
# 🎯 Distribuição de Dados Categóricos
# =======================
if show_distributions:
    st.subheader("Distribuição de Dados Categóricos")
    cols = ['gender', 'Partner', 'Dependents', 'InternetService', 'SeniorCitizen']
    cols_map = {
        'gender': 'Gênero',
        'Partner': 'Possui Parceiro',
        'Dependents': 'Dependentes',
        'InternetService': 'Tipo de Internet',
        'SeniorCitizen': 'Idoso (1=Sim, 0=Não)'
    }
    col1, col2, col3 = st.columns(3)

    for idx, col in enumerate(cols):
        data = df[col].value_counts().reset_index()
        data.columns = [col, 'count']  # Renomeia as colunas para garantir
        fig = px.bar(data, x=col, y='count', title=cols_map[col])
    
        if idx % 3 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif idx % 3 == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)
# =======================
# 🎯 Análise de Variáveis Numéricas
# =======================
if show_numeric:
    st.subheader("Análise de Variáveis Numéricas")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in numeric_cols:
        st.markdown(f"### 📊 {col}")

        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histograma de {col}")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.box(df, y=col, title=f"Boxplot de {col}")
            st.plotly_chart(fig2, use_container_width=True)


# =======================
# 🎯 Taxa de Cancelamento (Churn)
# =======================
if show_churn:
    st.subheader("Taxa de Cancelamento (Churn)")

    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']  # Melhor renomear para clareza

    fig = px.pie(
        churn_counts,
        names='Churn',
        values='Count',
        title='Distribuição de Churn',
        hole=0.4  # opcional: gráfico de donut
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Crosstab: Churn por Tipo de internet")
    crosstab_service = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    st.dataframe(crosstab_service.style.format("{:.2f}%"))


# =======================
# 🎯 Relações Bivariadas com Churn
# =======================
# =======================
# 🎯 Análise Bivariada com Churn
# =======================
if show_bivariadas:
    # ============================#

    st.title("Crosstab - Churn por Tipos de Serviço")

    # Lista de serviços a serem analisados
    services = ['PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # 🔍 Filtro de seleção
    selected_service = st.selectbox("Selecione o Tipo de Serviço:", services)
 
    # 🚩 Gerar Crosstab com Percentuais
    st.subheader(f"📊 Análise de Churn por {selected_service}")


    crosstab = pd.crosstab(
        df[selected_service],
        df['Churn'],
        margins=True,  # Inclui Total
        normalize='index'  # Percentual dentro de cada grupo do serviço
    ) * 100

    # 📑 Exibir Crosstab formatado
    st.dataframe(crosstab.style.format("{:.2f}%"))



    # 2. Churn por Idosos (SeniorCitizen)
    st.header("Churn por Idosos (Senior Citizen)")
    fig_senior = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group', title='Churn por Idosos')
    st.plotly_chart(fig_senior)

    # 3. Churn por Gênero
    st.header("Churn por Gênero")
    fig_gender = px.bar(df, x='gender', color='Churn', barmode='group', title='Churn por Gênero')
    st.plotly_chart(fig_gender)

    # 4. Churn por Tipo de Contrato
    #st.header("Churn por Tipo de Contrato")
    #fig_contract = px.histogram(df, x='ContractType', color='Churn', barmode='group', title='Churn por Tipo de Contrato')
    #st.plotly_chart(fig_contract)

    # 5. Churn por Método de Pagamento
    st.header("Churn por Método de Pagamento")
    fig_payment = px.histogram(df, x='PaymentMethod', color='Churn', barmode='group', title='Churn por Método de Pagamento')
    st.plotly_chart(fig_payment)

    
    


st.markdown("---")
st.caption("Dashboard desenvolvido com Streamlit | 📊 Senhor")
