import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Temperatura de Belém", layout="wide")
st.title("Projeto de Modelagem Estatística — Temperatura de Belém")
st.caption("Global Land Temperatures by City | Belém, Pará, Brasil")

@st.cache_data
def carregar():
    with zipfile.ZipFile("GlobalLandTemperaturesByCity.csv.zip") as z:
        nome=z.namelist()[0]
        partes=[]
        with z.open(nome) as f:
            for chunk in pd.read_csv(f,chunksize=300000):
                parte=chunk[(chunk["Country"]=="Brazil")&(chunk["City"]=="Belém")].copy()
                if not parte.empty: partes.append(parte)
    df=pd.concat(partes,ignore_index=True)
    df["dt"]=pd.to_datetime(df["dt"])
    df["year"]=df["dt"].dt.year
    df["month"]=df["dt"].dt.month
    return df.dropna(subset=["AverageTemperature"])

belem=carregar()
anual=belem.groupby("year",as_index=False).agg(AverageTemperature=("AverageTemperature","mean"),AverageTemperatureUncertainty=("AverageTemperatureUncertainty","mean"))
mensal=belem.groupby("month",as_index=False).agg(AverageTemperature=("AverageTemperature","mean"),AverageTemperatureUncertainty=("AverageTemperatureUncertainty","mean"))

st.sidebar.header("Filtros")
i,f=int(anual.year.min()),int(anual.year.max())
intervalo=st.sidebar.slider("Período",i,f,(i,f))
filtrado=anual[anual.year.between(intervalo[0],intervalo[1])]
reg=stats.linregress(filtrado.year,filtrado.AverageTemperature)
tc=stats.t.ppf(.975,len(filtrado)-2)

c1,c2,c3,c4=st.columns(4)
c1.metric("Anos",len(filtrado)); c2.metric("Média",f"{filtrado.AverageTemperature.mean():.2f} °C"); c3.metric("R²",f"{reg.rvalue**2:.3f}"); c4.metric("p-valor",f"{reg.pvalue:.2e}")

fig=px.line(filtrado,x="year",y="AverageTemperature",markers=True,labels={"year":"Ano","AverageTemperature":"Temperatura média anual (°C)"}); fig.update_layout(title="Temperatura média anual"); st.plotly_chart(fig,use_container_width=True)
fig2=px.line(mensal,x="month",y="AverageTemperature",markers=True,labels={"month":"Mês","AverageTemperature":"Temperatura média (°C)"}); fig2.update_layout(title="Temperatura média histórica por mês"); st.plotly_chart(fig2,use_container_width=True)

a,b=st.columns(2)
with a:
    st.subheader("Regressão linear")
    r=filtrado.copy(); r["Regressão"]=reg.intercept+reg.slope*r.year
    fig3=px.scatter(r,x="year",y="AverageTemperature",labels={"year":"Ano","AverageTemperature":"Temperatura média anual (°C)"}); fig3.add_scatter(x=r.year,y=r.Regressão,mode="lines",name="Regressão"); st.plotly_chart(fig3,use_container_width=True)
    st.dataframe(pd.DataFrame({"Métrica":["Inclinação","IC 95% inferior","IC 95% superior","Intercepto","R²","p-valor"],"Valor":[reg.slope,reg.slope-tc*reg.stderr,reg.slope+tc*reg.stderr,reg.intercept,reg.rvalue**2,reg.pvalue]}),hide_index=True)
with b:
    st.subheader("Teste entre períodos")
    inicial=anual[anual.year.between(1845,1894)].AverageTemperature; final=anual[anual.year.between(1964,2013)].AverageTemperature; teste=stats.ttest_ind(final,inicial,equal_var=False)
    st.dataframe(pd.DataFrame({"Grupo":["1845–1894","1964–2013"],"Média (°C)":[inicial.mean(),final.mean()],"Desvio padrão":[inicial.std(ddof=1),final.std(ddof=1)],"n":[len(inicial),len(final)]}),hide_index=True)
    st.metric("p-valor — t de Welch",f"{teste.pvalue:.2e}")

st.subheader("Extrapolação da tendência — 2014 a 2020")
rf=stats.linregress(anual.year,anual.AverageTemperature); futuro=pd.DataFrame({"year":range(2014,2021)}); futuro["Temperatura prevista (°C)"]=rf.intercept+rf.slope*futuro.year; st.dataframe(futuro,hide_index=True)
st.caption("A extrapolação é matemática e não representa uma previsão climática completa.")
