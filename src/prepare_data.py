"""Zona raw -> zona processed.

Run:  PYTHONHASHSEED=0 python -m src.prepare_data
"""
import hashlib

import numpy as np
import pandas as pd

from src.config import CODIGOS_FALTANTES, PROCESSED, RAW, TARGET


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_csv(RAW)                       # la zona raw solo se LEE
    n0 = len(df)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    if CODIGOS_FALTANTES:
        df = df.replace(CODIGOS_FALTANTES, np.nan)
    df = df.dropna(subset=[TARGET])             # sin objetivo no se puede entrenar
    # ---- Limpieza propia del dataset del grupo (agregar aquí) ----
    # Validaciones (no cambian los datos): las columnas configuradas existen y el objetivo es numérico.
    from src.config import COLS_CAT, COLS_NUM
    faltan = [c for c in COLS_NUM + COLS_CAT + [TARGET] if c not in df.columns]
    assert not faltan, f"columnas ausentes en raw: {faltan}"
    assert pd.api.types.is_numeric_dtype(df[TARGET]), "el objetivo debe ser numérico"

    # --------------------------------------------------------------
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"raw:       {n0} filas | md5 {md5(RAW)}")
    print(f"processed: {len(df)} filas | md5 {md5(PROCESSED)}")


if __name__ == "__main__":
    main()
