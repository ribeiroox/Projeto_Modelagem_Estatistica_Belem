import zipfile
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).parent
ZIP = PASTA / "GlobalLandTemperaturesByCity.csv.zip"
SAIDA = PASTA / "belem_serie_mensal.csv"

partes = []
with zipfile.ZipFile(ZIP) as z:
    with z.open(z.namelist()[0]) as f:
        for chunk in pd.read_csv(f, chunksize=300000):
            parte = chunk[(chunk["Country"] == "Brazil") & (chunk["City"] == "Belém")]
            if not parte.empty:
                partes.append(parte)

belem = pd.concat(partes, ignore_index=True)
belem["dt"] = pd.to_datetime(belem["dt"])
belem = belem[["dt", "AverageTemperature", "AverageTemperatureUncertainty"]].sort_values("dt")
belem.to_csv(SAIDA, index=False)
print(len(belem), "linhas |", belem["AverageTemperature"].isna().sum(), "sem temperatura |", belem["dt"].min().date(), "a", belem["dt"].max().date())
