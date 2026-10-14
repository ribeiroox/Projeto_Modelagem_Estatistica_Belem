import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Temperatura de Belém", layout="wide")
st.title("Projeto de Modelagem Estatística — Temperatura de Belém")
st.caption("Global Land Temperatures by City | Belém, Pará, Brasil")

PASTA = Path(__file__).parent
ARQUIVO = PASTA / "belem_serie_mensal.csv"
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
VARIAVEIS = {"temperatura": "Temperatura média anual (°C)", "anomalia": "Anomalia anual (°C)"}
PARAMETROS = inspect.signature(st.plotly_chart).parameters
LARGURA = {"width": "stretch"} if "width" in PARAMETROS else {"use_container_width": True}
COR_FAIXA = "rgba(99,110,250,0.20)"
COR_FAIXA_PREVISAO = "rgba(239,85,59,0.15)"


def mostrar(fig, chave):
    st.plotly_chart(fig, key=chave, **LARGURA)


@st.cache_data
def carregar():
    df = pd.read_csv(ARQUIVO, parse_dates=["dt"])
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    media_mes = df.groupby("month")["AverageTemperature"].transform("mean")
    df["anomalia"] = df["AverageTemperature"] - media_mes
    df["estacao"] = np.where(df["month"].isin([12, 1, 2, 3, 4, 5]), "Dez–Mai", "Jun–Nov")
    df["mes_nome"] = df["month"].map(lambda m: MESES[m - 1])
    return df


def agregar(base, minimo):
    valido = base.dropna(subset=["AverageTemperature"])
    g = valido.groupby("year").agg(
        temperatura=("AverageTemperature", "mean"),
        anomalia=("anomalia", "mean"),
        incerteza=("AverageTemperatureUncertainty", "mean"),
        meses_validos=("AverageTemperature", "size"),
    ).reset_index()
    g = g[g["meses_validos"] >= minimo].copy()
    g["decada"] = (g["year"] // 10) * 10
    return g


def regressao(x, y):
    reg = stats.linregress(x, y)
    res = y - (reg.intercept + reg.slope * x)
    s = np.sqrt(np.sum(res ** 2) / (len(x) - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    return reg, res, s, sxx


def faixas(xn, x, s, sxx, tc):
    h = 1 / len(x) + (xn - x.mean()) ** 2 / sxx
    return tc * s * np.sqrt(h), tc * s * np.sqrt(1 + h)


def adicionar_faixa(fig, x, inf, sup, nome, cor):
    fig.add_scatter(x=x, y=sup, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=inf, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=cor, name=nome, hoverinfo="skip")


def fmt(v):
    return f"{v:.6g}"


if not ARQUIVO.exists():
    st.error("Arquivo belem_serie_mensal.csv não encontrado na pasta do app. Execute preparar_dados.py.")
    st.stop()

dados = carregar()
validos = dados.dropna(subset=["AverageTemperature"])
ini, fim = int(validos["year"].min()), int(validos["year"].max())

st.sidebar.header("Filtros")
variavel = st.sidebar.radio("Variável analisada", list(VARIAVEIS), format_func=VARIAVEIS.get)
periodo = st.sidebar.slider("Período", ini, fim, (ini, fim))
meses = st.sidebar.multiselect("Meses considerados", list(range(1, 13)), default=list(range(1, 13)), format_func=lambda m: MESES[m - 1])
if not meses:
    st.sidebar.error("Selecione ao menos um mês.")
    st.stop()
minimo = st.sidebar.number_input("Mínimo de meses válidos por ano", min_value=1, max_value=len(meses), value=1, step=1)
nivel = st.sidebar.select_slider("Nível de confiança", options=[0.90, 0.95, 0.99], value=0.95, format_func=lambda v: f"{v:.0%}")
alfa = 1 - nivel

rotulo = VARIAVEIS[variavel]
base = dados[dados["month"].isin(meses)]
anual = agregar(base, int(minimo))
filtrado = anual[anual["year"].between(*periodo)].reset_index(drop=True)

if len(filtrado) < 10:
    st.sidebar.error("O recorte atual tem menos de 10 anos. Amplie o período ou reduza o mínimo de meses válidos.")
    st.stop()

x = filtrado["year"].to_numpy(float)
y = filtrado[variavel].to_numpy(float)
n = len(x)
gl = n - 2
reg, res, s, sxx = regressao(x, y)
tc = stats.t.ppf(1 - alfa / 2, gl)
t_obs = reg.slope / reg.stderr
r2 = reg.rvalue ** 2
r2_aj = 1 - (1 - r2) * (n - 1) / gl
dw = np.sum(np.diff(res) ** 2) / np.sum(res ** 2)
shapiro_res = stats.shapiro(res)

aba1, aba2, aba3, aba4, aba5 = st.tabs(["Exploração", "Qualidade dos dados", "Regressão", "Testes de hipótese", "Desempenho e previsão"])

with aba1:
    c = st.columns(5)
    c[0].metric("Anos", n)
    c[1].metric("Média", f"{y.mean():.2f} °C")
    c[2].metric("Desvio padrão", f"{y.std(ddof=1):.3f} °C")
    c[3].metric("Mínimo", f"{y.min():.2f} °C")
    c[4].metric("Máximo", f"{y.max():.2f} °C")

    o1, o2 = st.columns(2)
    exibir_incerteza = o1.checkbox("Exibir incerteza do dataset como barras de erro", value=False)
    janela = o2.slider("Janela da média móvel (anos)", 1, 15, 1)

    fig = px.line(filtrado, x="year", y=variavel, markers=True, error_y="incerteza" if exibir_incerteza else None,
                labels={"year": "Ano", variavel: rotulo})
    fig.update_layout(title="Série anual")
    if janela > 1:
        fig.add_scatter(x=filtrado["year"], y=filtrado[variavel].rolling(janela, center=True).mean(),
                        mode="lines", name=f"Média móvel ({janela} anos)", line=dict(width=4))
    mostrar(fig, "serie_anual")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.box(filtrado, x="decada", y=variavel, points="all", labels={"decada": "Década", variavel: rotulo})
        fig.update_layout(title="Distribuição por década")
        mostrar(fig, "box_decada")
    with g2:
        classes = st.slider("Número de classes do histograma", 5, 40, 15)
        fig = px.histogram(filtrado, x=variavel, nbins=classes, labels={variavel: rotulo})
        fig.update_layout(title="Histograma", yaxis_title="Frequência")
        mostrar(fig, "histograma")

    mensal = base[base["year"].between(*periodo)].dropna(subset=["AverageTemperature"])
    fig = px.box(mensal, x="mes_nome", y="AverageTemperature", color="estacao",
                category_orders={"mes_nome": [MESES[m - 1] for m in sorted(meses)]},
                labels={"mes_nome": "Mês", "AverageTemperature": "Temperatura mensal (°C)", "estacao": "Meses"})
    fig.update_layout(title="Distribuição mensal no período")
    mostrar(fig, "box_mensal")

with aba2:
    periodo_df = dados[dados["year"].between(*periodo)]
    contagem = periodo_df.dropna(subset=["AverageTemperature"]).groupby("year").size().rename("validos")
    qualidade = pd.DataFrame({"year": range(periodo[0], periodo[1] + 1)}).merge(contagem, on="year", how="left")
    qualidade["validos"] = qualidade["validos"].fillna(0).astype(int)
    qualidade["ausentes"] = 12 - qualidade["validos"]
    esperado = 12 * len(qualidade)

    c = st.columns(4)
    c[0].metric("Meses esperados", esperado)
    c[1].metric("Meses ausentes", int(qualidade["ausentes"].sum()))
    c[2].metric("% ausente", f"{100 * qualidade['ausentes'].sum() / esperado:.2f}%")
    c[3].metric("Anos excluídos pelo filtro", len(qualidade) - n)

    q1, q2 = st.columns(2)
    with q1:
        fig = px.bar(qualidade, x="year", y="ausentes", labels={"year": "Ano", "ausentes": "Meses ausentes"})
        fig.update_layout(title="Meses ausentes por ano")
        mostrar(fig, "ausentes")
    with q2:
        fig = px.line(filtrado, x="year", y="incerteza", markers=True,
                    labels={"year": "Ano", "incerteza": "Incerteza média (°C)"})
        fig.update_layout(title="Incerteza média reportada por ano")
        mostrar(fig, "incerteza")

    tabela_dados = filtrado.rename(columns={"year": "Ano", "temperatura": "Temperatura (°C)", "anomalia": "Anomalia (°C)",
                                            "incerteza": "Incerteza (°C)", "meses_validos": "Meses válidos", "decada": "Década"})
    st.dataframe(tabela_dados, hide_index=True)
    st.download_button("Baixar tabela anual (CSV)", tabela_dados.to_csv(index=False).encode("utf-8"), "serie_anual.csv", "text/csv")

with aba3:
    c = st.columns(4)
    c[0].metric("Inclinação (°C/década)", f"{10 * reg.slope:.3f}")
    c[1].metric("R²", f"{r2:.3f}")
    c[2].metric("R² ajustado", f"{r2_aj:.3f}")
    c[3].metric("p-valor da inclinação", f"{reg.pvalue:.2e}")

    faixas_sel = st.multiselect("Faixas no gráfico", ["Intervalo de confiança da média", "Intervalo de previsão"],
                                default=["Intervalo de confiança da média"])
    xg = np.linspace(x.min(), x.max(), 200)
    yg = reg.intercept + reg.slope * xg
    conf, pred = faixas(xg, x, s, sxx, tc)
    fig = px.scatter(filtrado, x="year", y=variavel, labels={"year": "Ano", variavel: rotulo})
    if "Intervalo de previsão" in faixas_sel:
        adicionar_faixa(fig, xg, yg - pred, yg + pred, f"Previsão {nivel:.0%}", COR_FAIXA_PREVISAO)
    if "Intervalo de confiança da média" in faixas_sel:
        adicionar_faixa(fig, xg, yg - conf, yg + conf, f"Confiança {nivel:.0%}", COR_FAIXA)
    fig.add_scatter(x=xg, y=yg, mode="lines", name="Regressão", line=dict(width=3))
    fig.update_layout(title="Regressão linear simples")
    mostrar(fig, "regressao")

    estatisticas = {
        "Inclinação β₁ (°C/ano)": reg.slope,
        "Erro padrão de β₁": reg.stderr,
        f"IC {nivel:.0%} de β₁ — inferior": reg.slope - tc * reg.stderr,
        f"IC {nivel:.0%} de β₁ — superior": reg.slope + tc * reg.stderr,
        "Intercepto β₀": reg.intercept,
        "R²": r2,
        "R² ajustado": r2_aj,
        "Erro padrão residual (s)": s,
        "Estatística t (H₀: β₁ = 0)": t_obs,
        "Graus de liberdade (n − 2)": gl,
        f"t crítico bilateral ({nivel:.0%})": tc,
        "p-valor": reg.pvalue,
        "Durbin-Watson": dw,
        "Shapiro-Wilk dos resíduos (p-valor)": shapiro_res.pvalue,
    }
    st.dataframe(pd.DataFrame({"Estatística": list(estatisticas), "Valor": [fmt(v) for v in estatisticas.values()]}), hide_index=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        fig = px.scatter(x=x, y=res, labels={"x": "Ano", "y": "Resíduo (°C)"})
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title="Resíduos × ano")
        mostrar(fig, "res_ano")
    with d2:
        fig = px.histogram(x=res, nbins=15, labels={"x": "Resíduo (°C)"})
        fig.update_layout(title="Histograma dos resíduos", yaxis_title="Frequência")
        mostrar(fig, "res_hist")
    with d3:
        (teoricos, ordenados), (a_qq, b_qq, _) = stats.probplot(res, dist="norm")
        fig = px.scatter(x=teoricos, y=ordenados, labels={"x": "Quantis teóricos", "y": "Quantis dos resíduos"})
        fig.add_scatter(x=teoricos, y=a_qq * teoricos + b_qq, mode="lines", name="Referência", showlegend=False)
        fig.update_layout(title="QQ-plot dos resíduos")
        mostrar(fig, "res_qq")

with aba4:
    lo, hi = int(anual["year"].min()), int(anual["year"].max())
    k1, k2 = st.columns(2)
    pa = k1.slider("Grupo A", lo, hi, (lo, min(lo + 49, hi)))
    pb = k2.slider("Grupo B", lo, hi, (max(hi - 49, lo), hi))
    opcoes = {"Bilateral (μB ≠ μA)": "two-sided", "Unilateral (μB > μA)": "greater", "Unilateral (μB < μA)": "less"}
    alternativa = opcoes[st.radio("Hipótese alternativa", list(opcoes), horizontal=True)]

    ga = anual[anual["year"].between(*pa)][variavel].to_numpy(float)
    gb = anual[anual["year"].between(*pb)][variavel].to_numpy(float)

    if pa[1] >= pb[0]:
        st.error("Os grupos não podem se sobrepor: o fim do grupo A deve ser anterior ao início do grupo B.")
    elif len(ga) < 5 or len(gb) < 5:
        st.error("Cada grupo precisa ter ao menos 5 anos válidos.")
    else:
        na, nb = len(ga), len(gb)
        va, vb = ga.var(ddof=1), gb.var(ddof=1)
        ep = np.sqrt(va / na + vb / nb)
        gl_w = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        dif = gb.mean() - ga.mean()
        welch = stats.ttest_ind(gb, ga, equal_var=False, alternative=alternativa)
        t_ic = stats.t.ppf(1 - alfa / 2, gl_w)
        sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        mw = stats.mannwhitneyu(gb, ga, alternative=alternativa)
        levene = stats.levene(ga, gb)

        if alternativa == "two-sided":
            crit = stats.t.ppf(1 - alfa / 2, gl_w)
            texto_crit = f"±{crit:.4f}"
        else:
            crit = stats.t.ppf(1 - alfa, gl_w)
            texto_crit = f"{crit:.4f}" if alternativa == "greater" else f"-{crit:.4f}"

        rotulo_a = f"A ({pa[0]}–{pa[1]})"
        rotulo_b = f"B ({pb[0]}–{pb[1]})"

        st.dataframe(pd.DataFrame({
            "Grupo": [rotulo_a, rotulo_b],
            "Média": [ga.mean(), gb.mean()],
            "Desvio padrão": [np.sqrt(va), np.sqrt(vb)],
            "Variância": [va, vb],
            "n": [na, nb],
        }), hide_index=True)

        resultado = {
            "Diferença de médias (B − A)": dif,
            "Erro padrão da diferença": ep,
            "Estatística t de Welch": welch.statistic,
            "Graus de liberdade (Welch–Satterthwaite)": gl_w,
            "t crítico": texto_crit,
            "p-valor": welch.pvalue,
            f"IC {nivel:.0%} da diferença — inferior": dif - t_ic * ep,
            f"IC {nivel:.0%} da diferença — superior": dif + t_ic * ep,
            "d de Cohen": dif / sp,
        }
        pressupostos = {
            "Shapiro-Wilk grupo A (p-valor)": stats.shapiro(ga).pvalue,
            "Shapiro-Wilk grupo B (p-valor)": stats.shapiro(gb).pvalue,
            "Levene — igualdade de variâncias (p-valor)": levene.pvalue,
            "Mann-Whitney U (p-valor)": mw.pvalue,
        }
        m1, m2 = st.columns(2)
        with m1:
            st.subheader("Teste t de Welch")
            st.dataframe(pd.DataFrame({"Estatística": list(resultado), "Valor": [v if isinstance(v, str) else fmt(v) for v in resultado.values()]}), hide_index=True)
        with m2:
            st.subheader("Pressupostos e teste não paramétrico")
            st.dataframe(pd.DataFrame({"Estatística": list(pressupostos), "Valor": [fmt(v) for v in pressupostos.values()]}), hide_index=True)

        p1, p2 = st.columns(2)
        with p1:
            comparacao = pd.DataFrame({"Grupo": [rotulo_a] * na + [rotulo_b] * nb, rotulo: np.concatenate([ga, gb])})
            fig = px.box(comparacao, x="Grupo", y=rotulo, points="all")
            fig.update_layout(title="Comparação entre grupos")
            mostrar(fig, "box_grupos")
        with p2:
            xs = np.linspace(-max(6, abs(welch.statistic) * 1.2), max(6, abs(welch.statistic) * 1.2), 600)
            dens = stats.t.pdf(xs, gl_w)
            if alternativa == "two-sided":
                cauda = (xs <= -crit) | (xs >= crit)
            elif alternativa == "greater":
                cauda = xs >= crit
            else:
                cauda = xs <= -crit
            fig = px.line(x=xs, y=dens, labels={"x": "t", "y": "Densidade"})
            fig.add_scatter(x=xs[cauda], y=dens[cauda], mode="markers", name="Região de rejeição", marker=dict(size=4))
            fig.add_vline(x=welch.statistic, line_dash="dash", annotation_text="t observado")
            fig.update_layout(title=f"Distribuição t com {gl_w:.1f} graus de liberdade")
            mostrar(fig, "dist_t")

with aba5:
    lo_c, hi_c = int(filtrado["year"].min()) + 9, int(filtrado["year"].max()) - 2
    if lo_c >= hi_c:
        st.warning("O recorte atual é curto demais para separar treino e teste.")
    else:
        corte = st.slider("Último ano do treino", lo_c, hi_c, int(np.clip(1999, lo_c, hi_c)))
        treino = filtrado[filtrado["year"] <= corte]
        teste = filtrado[filtrado["year"] > corte]
        if len(treino) < 5 or len(teste) < 2:
            st.warning("Treino ou teste com poucas observações para o ano de corte escolhido.")
        else:
            xt, yt = treino["year"].to_numpy(float), treino[variavel].to_numpy(float)
            xe, ye = teste["year"].to_numpy(float), teste[variavel].to_numpy(float)
            reg_t, _, s_t, sxx_t = regressao(xt, yt)
            tc_t = stats.t.ppf(1 - alfa / 2, len(xt) - 2)
            ye_prev = reg_t.intercept + reg_t.slope * xe
            _, pi_t = faixas(xe, xt, s_t, sxx_t, tc_t)
            rmse = np.sqrt(np.mean((ye - ye_prev) ** 2))
            mae = np.mean(np.abs(ye - ye_prev))
            rmse_base = np.sqrt(np.mean((ye - yt.mean()) ** 2))
            ganho = 1 - np.sum((ye - ye_prev) ** 2) / np.sum((ye - yt.mean()) ** 2)
            cobertura = np.mean(np.abs(ye - ye_prev) <= pi_t)

            c = st.columns(6)
            c[0].metric("RMSE", f"{rmse:.3f}")
            c[1].metric("MAE", f"{mae:.3f}")
            c[2].metric("RMSE baseline", f"{rmse_base:.3f}")
            c[3].metric("Ganho sobre baseline", f"{ganho:.3f}")
            c[4].metric("Cobertura do IP", f"{cobertura:.0%}")
            c[5].metric("Treino / teste", f"{len(xt)} / {len(xe)}")

            fig = px.line(filtrado, x="year", y=variavel, labels={"year": "Ano", variavel: rotulo})
            fig.update_traces(name="Observado", showlegend=True)
            adicionar_faixa(fig, xe, ye_prev - pi_t, ye_prev + pi_t, f"Previsão {nivel:.0%}", COR_FAIXA_PREVISAO)
            fig.add_scatter(x=xe, y=ye_prev, mode="lines+markers", name="Previsto (teste)", line=dict(width=3))
            fig.add_vline(x=corte + 0.5, line_dash="dash", annotation_text="corte")
            fig.update_layout(title="Avaliação temporal")
            mostrar(fig, "avaliacao")

    horizonte = st.slider("Horizonte da previsão (anos)", 1, 20, 7)
    xf = np.arange(x.max() + 1, x.max() + 1 + horizonte)
    yf = reg.intercept + reg.slope * xf
    _, pi_f = faixas(xf, x, s, sxx, tc)
    previsao = pd.DataFrame({"Ano": xf.astype(int), "Previsto": yf, "Limite inferior": yf - pi_f, "Limite superior": yf + pi_f})

    xr = np.concatenate([x, xf])
    fig = px.line(filtrado, x="year", y=variavel, labels={"year": "Ano", variavel: rotulo})
    fig.update_traces(name="Observado", showlegend=True)
    adicionar_faixa(fig, xf, previsao["Limite inferior"], previsao["Limite superior"], f"Previsão {nivel:.0%}", COR_FAIXA_PREVISAO)
    fig.add_scatter(x=xr, y=reg.intercept + reg.slope * xr, mode="lines", name="Tendência ajustada")
    fig.add_scatter(x=xf, y=yf, mode="markers", name="Previsto", marker=dict(size=8))
    fig.update_layout(title="Extrapolação da tendência")
    mostrar(fig, "previsao")

    st.dataframe(previsao.round(3), hide_index=True)
    st.download_button("Baixar previsão (CSV)", previsao.to_csv(index=False).encode("utf-8"), "previsao.csv", "text/csv")
