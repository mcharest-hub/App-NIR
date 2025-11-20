# -*- coding: utf-8 -*-
# ============================================================
#  App NIR — Dashboard Projet Argos-Signature (Streamlit)
# ============================================================

import os
import re
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

# ------------------------------------------------------------
# Constantes / Config
# ------------------------------------------------------------

CARTO_BASEMAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_RASTER_URL = (
    "https://c.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png"
)

PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

# Regroupements de substances (utilisés dans Statistiques)
SUBSTANCE_GROUPS = {
    # Amphétamines
    "Amphétamine en poudre": "Amphétamine",
    "pâte d'amphétamine": "Amphétamine",
    "cristaux d'amphétamine": "Amphétamine",
    "Cristaux d'amphétamine": "Amphétamine",
    # Méthamphétamines
    "Métamphétamine en poudre": "Méthamphétamine",
    "Cristaux de Méthamphétamine": "Méthamphétamine",
    # MDMA
    "Ecstasy (MDMA)": "MDMA",
    "Cristaux de MDMA": "MDMA",
    # CBD
    "Hashish CBD": "Résine CBD",
    # Inconnue
    "substance inconnue": "Substance inconnue",
    # Tobacco
    "Tobacco": "Tabac",
}


# ------------------------------------------------------------
# Carto — Basemap sans token (fallback)
# ------------------------------------------------------------
def deck_with_fallback(layers, view_state, tooltip=None):
    """
    Construit un pdk.Deck avec Carto GL ; si cela échoue (ex. sandbox Streamlit Cloud),
    on ajoute un TileLayer raster et on désactive map_style (aucun token requis).
    """
    try:
        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=CARTO_BASEMAP,
            tooltip=tooltip,
        )
    except Exception:
        tile_layer = pdk.Layer(
            "TileLayer",
            data=CARTO_RASTER_URL,
            minZoom=0,
            maxZoom=19,
            tileSize=256,
        )
        return pdk.Deck(
            layers=[tile_layer] + layers,
            initial_view_state=view_state,
            map_style=None,
            tooltip=tooltip,
        )


# ------------------------------------------------------------
# Préprocessing optionnel (module local)
# ------------------------------------------------------------
sys.path.append(r"C:\Users\mcharest\Documents\Doctorat\python_scripts")
try:
    from preprocessing import apply_snv, apply_savgol
except Exception as e:
    apply_snv = apply_savgol = None
    st.warning(f"Module preprocessing introuvable : {e}")


# ------------------------------------------------------------
# Style / Thème Altair
# ------------------------------------------------------------
def _register_clean_theme() -> None:
    theme = {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFontSize": 13,
                "titleFontSize": 14,
                "labelColor": "#111827",
                "titleColor": "#111827",
                "grid": True,
                "gridColor": "#E5E7EB",
                "domain": False,
                "tickColor": "#9CA3AF",
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 12,
                "labelColor": "#111827",
                "titleColor": "#111827",
            },
            "title": {"fontSize": 20, "color": "#111827", "font": "Inter"},
            "range": {"category": PALETTE},
        }
    }
    alt.themes.register("clean_tableau20", lambda: theme)
    alt.themes.enable("clean_tableau20")


st.set_page_config(
    page_title="Dashboard Argos-Signature", page_icon="📊", layout="wide"
)
_register_clean_theme()
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.streamlit-expanderHeader { font-weight: 600; }
.stRadio > label, .stSelectbox > label { font-weight: 600; }
.kpi-card { padding: 1rem; border: 1px solid #e5e7eb; border-radius: .75rem; background: #fff; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #111827; }
.kpi-label { color: #6b7280; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helpers (I/O, colonnes, couleurs, dates…)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


@st.cache_data(show_spinner=False)
def _read_excel(file, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, list]:
    xls = pd.ExcelFile(file)
    sheets = xls.sheet_names
    df = pd.read_excel(xls, sheet_name=sheet_name or sheets[0])
    return df, sheets


def _ext(name: str) -> str:
    return (name.split(".")[-1] if "." in name else "").lower()


def _find_col(candidates, cols) -> Optional[str]:
    cols_l = [c.lower() for c in cols]
    for c in candidates:
        if c.lower() in cols_l:
            return cols[cols_l.index(c.lower())]
    return None


def _parse_dates_series(s: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(s, errors="coerce")
    s2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return s1 if s1.notna().sum() >= s2.notna().sum() else s2


def _melt_regressors(df: pd.DataFrame) -> pd.DataFrame:
    """Met au format long les paires regressors_*_value_type / regressors_*_value_numeric."""
    cols = df.columns
    idxs = []
    for c in cols:
        m = re.match(r"regressors_(\d+)_value_type$", c)
        if m and f"regressors_{m.group(1)}_value_numeric" in cols:
            idxs.append(m.group(1))
    if not idxs:
        return pd.DataFrame()

    sample_col = _find_col(["sample_name"], cols)
    date_col = _find_col(["seizure_date"], cols)
    substance_col = _find_col(["substance", "substance_name", "drug", "name_x"], cols)

    parts = []
    for i in idxs:
        tcol, vcol = f"regressors_{i}_value_type", f"regressors_{i}_value_numeric"
        keep = [tcol, vcol] + [c for c in (sample_col, date_col, substance_col) if c]
        part = df[keep].copy()
        part.rename(
            columns={
                tcol: "purity_type",
                vcol: "purity_value",
                **({sample_col: "sample_name"} if sample_col else {}),
                **({date_col: "seizure_date"} if date_col else {}),
                **({substance_col: "substance"} if substance_col else {}),
            },
            inplace=True,
        )
        parts.append(part)

    long = pd.concat(parts, ignore_index=True)
    long["purity_type_raw"] = long["purity_type"].astype(str)
    long["purity_type_norm"] = (
        long["purity_type_raw"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )
    mapping = (
        long.dropna(subset=["purity_type_raw"])
        .drop_duplicates("purity_type_norm")
        .set_index("purity_type_norm")["purity_type_raw"]
        .to_dict()
    )
    long["purity_type_display"] = long["purity_type_norm"].map(mapping)
    return long


def _find_geo_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    return _find_col(["latitude", "lat"], cols), _find_col(
        ["longitude", "lon", "lng"], cols
    )


def _get_spectrum_cols(df: pd.DataFrame):
    """
    Détecte les colonnes spectrales de la forme 'spectrum_XXXX' et extrait les longueurs d'onde.
    Retourne (liste_colonnes, série_wavelengths indexée par colonne).
    """
    if df is None or df.empty:
        return [], pd.Series(dtype=float)

    spec_cols = [
        c for c in df.columns if isinstance(c, str) and c.startswith("spectrum_")
    ]
    if not spec_cols:
        return [], pd.Series(dtype=float)

    def _extract_wl(col: str) -> Optional[float]:
        s = str(col).replace("spectrum_", "")
        try:
            return float(s)
        except Exception:
            m = re.search(r"(\d+(\.\d+)?)", s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

    wls_dict = {c: _extract_wl(c) for c in spec_cols}
    wls = pd.Series(wls_dict, dtype="float64").dropna()

    # garder seulement les colonnes pour lesquelles on a une longueur d’onde valide
    spec_cols = [c for c in spec_cols if c in wls.index]

    return spec_cols, wls


def _ensure_color_map(keys, stored=None, palette=PALETTE):
    stored = stored or {}
    out, i = {}, 0
    for k in keys:
        if k in stored:
            out[k] = stored[k]
        else:
            out[k] = palette[i % len(palette)]
            i += 1
    return out


def _find_weight_col(df: pd.DataFrame) -> Optional[str]:
    """Détecte la colonne de poids net (priorise net_* / poids_*)."""
    return _find_col(
        [
            "net_weight",
            "weight_net",
            "netweight",
            "poids_net",
            "poids (net)",
            "poids_net_g",
            "weight",
            "poids",
        ],
        df.columns.tolist(),
    )


def _hex_to_rgb(hx, _default="#2563EB"):
    if hx is None or isinstance(hx, float):
        hx = _default
    hx = str(hx).strip()
    if not hx or hx.lower() == "nan":
        hx = _default
    if not hx.startswith("#"):
        hx = "#" + hx
    if len(hx) == 4:
        hx = "#" + "".join([ch * 2 for ch in hx[1:]])
    try:
        return [int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)]
    except Exception:
        hx = _default
        return [int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)]


def _hex_to_rgba(hx, alpha=180):
    return _hex_to_rgb(hx) + [alpha]

def replace_spectra(df_in: pd.DataFrame, new_spec: pd.DataFrame, prefix: str = "spectrum_") -> pd.DataFrame:
    """
    Remplace les colonnes spectrales (prefix, par défaut 'spectrum_') de df_in par new_spec,
    en conservant toutes les autres colonnes.
    """
    non_spec = [
        c
        for c in df_in.columns
        if not (isinstance(c, str) and c.startswith(prefix))
    ]
    return pd.concat(
        [df_in[non_spec].reset_index(drop=True), new_spec.reset_index(drop=True)],
        axis=1,
    )


def run_spectral_pipeline(
    df_in: pd.DataFrame,
    pipeline: list,
    cut_range: tuple[int, int],
    sg_window: int,
    sg_poly: int,
    *,
    prefix: str = "spectrum_",
) -> pd.DataFrame:
    """
    Applique, dans l'ordre, les étapes du pipeline sur les colonnes spectrum_* :
    - 'Découpe'
    - 'SG dérivée 2'
    - 'SNV'

    Comportement identique à ce que tu avais déjà, mais factorisé.
    """
    working = df_in.copy()

    if not pipeline:
        return working

    for step in pipeline:
        spec_cols_now, wls_now = _get_spectrum_cols(working)
        if not spec_cols_now:
            st.error("Plus aucune colonne spectrale détectée pendant le pipeline.")
            st.stop()

        spec = working[spec_cols_now].copy()

        if step == "Découpe":
            wl_lo, wl_hi = cut_range
            keep_cols = [
                c
                for c in spec_cols_now
                if (wls_now[c] >= wl_lo and wls_now[c] <= wl_hi)
            ]
            if not keep_cols:
                st.error("Découpe trop restrictive : aucune colonne conservée.")
                st.stop()

            working = replace_spectra(working, spec[keep_cols], prefix=prefix)
            st.caption(f"✔️ Découpe {wl_lo}–{wl_hi} nm ({len(keep_cols)} colonnes).")

        elif step == "SG dérivée 2":
            try:
                spec = apply_savgol(
                    spec, window_length=int(sg_window), polyorder=int(sg_poly), deriv=2
                )
                # on conserve le même ordre de colonnes qu'avant
                spec.columns = spec_cols_now[: spec.shape[1]]
                working = replace_spectra(working, spec, prefix=prefix)
                st.caption(f"✔️ SG dérivée 2 (fenêtre={sg_window}, ordre={sg_poly}).")
            except Exception as e:
                st.error(f"Erreur Savitzky–Golay : {e}")
                st.stop()

        elif step == "SNV":
            try:
                spec = apply_snv(spec)
                spec.columns = spec_cols_now[: spec.shape[1]]
                working = replace_spectra(working, spec, prefix=prefix)
                st.caption("✔️ SNV appliqué.")
            except Exception as e:
                st.error(f"Erreur SNV : {e}")
                st.stop()

    return working


# ------------------------------------------------------------
# Section réutilisable : Détail d’un cluster
# ------------------------------------------------------------
def render_cluster_detail_section(payload: Optional[dict] = None) -> None:
    """
    Affiche la section 'Détail d’un cluster' à partir d’un payload (ou session).
    Le payload doit contenir : cluster_df, cluster_df_full, sample_col, out, df_raw, date_col_raw, namex_col_raw.
    """
    st.markdown("---")
    st.subheader("Détail d’un cluster")

    payload = payload or st.session_state.get("cluster_detail")
    if not payload:
        st.info(
            "Générez d’abord des clusters dans l’onglet Profilage (aucun détail disponible)."
        )
        return

    cluster_df = payload.get("cluster_df")
    cluster_df_full = payload.get("cluster_df_full")
    sample_col = payload.get("sample_col")
    out = payload.get("out")
    df_raw = payload.get("df_raw")
    date_col_raw = payload.get("date_col_raw")
    namex_col_raw = payload.get("namex_col_raw")

    if cluster_df is None or cluster_df_full is None or sample_col is None:
        st.info("Payload de clustering incomplet — impossible d’afficher le détail.")
        return

    sizes_map = (
        cluster_df_full.groupby("cluster")["label"].count().rename("taille").to_dict()
    )
    cluster_ids = sorted(sizes_map.keys())

    def _label_cluster(cid: int) -> str:
        return f"Cluster {cid} — {sizes_map.get(cid, 0)} échantillons"

    chosen_cluster = st.selectbox(
        "Choisir un cluster à explorer",
        options=cluster_ids,
        format_func=_label_cluster,
        key="cluster_detail_select",
    )

    # Échantillons de ce cluster
    sample_list = (
        cluster_df.loc[cluster_df["cluster"] == int(chosen_cluster), "label"]
        .astype(str)
        .tolist()
    )
    if not sample_list:
        st.info("Ce cluster ne contient aucun échantillon sélectionnable.")
        return

    # Sous-ensemble brut
    if df_raw is not None and sample_col in df_raw.columns:
        subset_raw = df_raw[df_raw[sample_col].astype(str).isin(sample_list)].copy()
    else:
        subset_raw = (
            out[out[sample_col].astype(str).isin(sample_list)].copy()
            if (out is not None and sample_col in getattr(out, "columns", []))
            else pd.DataFrame()
        )

    # Onglets
    tab_sum, tab_purity, tab_time, tab_map = st.tabs(
        ["Résumé", "Pureté", "Temporalité", "Carte"]
    )

    # ----- Résumé
    with tab_sum:
        st.markdown(f"**Cluster {chosen_cluster}** — {_label_cluster(chosen_cluster)}")

        if (
            (not subset_raw.empty)
            and date_col_raw
            and (date_col_raw in subset_raw.columns)
        ):
            subset_raw[date_col_raw] = _parse_dates_series(subset_raw[date_col_raw])
            dmin = pd.to_datetime(subset_raw[date_col_raw]).min()
            dmax = pd.to_datetime(subset_raw[date_col_raw]).max()
            dstr = lambda d: (d.strftime("%d/%m/%Y") if pd.notna(d) else "N/A")
            period_label = f"{dstr(dmin)} – {dstr(dmax)}"
        else:
            period_label = "N/A – N/A"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Échantillons</div>'
                f'<div class="kpi-value">{len(sample_list):,}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Période</div>'
                f'<div class="kpi-value">{period_label}</div></div>',
                unsafe_allow_html=True,
            )

        # Tableau récap — sample / Pureté moyenne / Date / Poids net
        with st.expander("Liste des échantillons du cluster", expanded=True):
            base_samples = pd.DataFrame(
                {sample_col: pd.Series(sample_list, dtype=str)}
            ).drop_duplicates()

            # Date (priorité df_raw)
            if (
                (df_raw is not None)
                and date_col_raw
                and (date_col_raw in df_raw.columns)
            ):
                dsub = df_raw[[sample_col, date_col_raw]].copy()
                dsub[sample_col] = dsub[sample_col].astype(str)
                dsub[date_col_raw] = _parse_dates_series(dsub[date_col_raw])
                last_date_raw = (
                    dsub.dropna(subset=[date_col_raw])
                    .groupby(sample_col, as_index=False)[date_col_raw]
                    .max()
                    .rename(columns={date_col_raw: "Date"})
                )
                last_date_raw["Date"] = pd.to_datetime(
                    last_date_raw["Date"]
                ).dt.strftime("%d/%m/%Y")
            else:
                last_date_raw = pd.DataFrame(columns=[sample_col, "Date"])

            # Pureté moyenne (regressors_*)
            pure_long_all = (
                _melt_regressors(df_raw) if df_raw is not None else pd.DataFrame()
            )
            if not pure_long_all.empty and (sample_col in pure_long_all.columns):
                pure_long_all = pure_long_all.copy()
                pure_long_all[sample_col] = pure_long_all[sample_col].astype(str)
                pure_sub_tbl = pure_long_all[
                    pure_long_all[sample_col].isin(base_samples[sample_col])
                ].copy()
                pure_sub_tbl = pure_sub_tbl.dropna(subset=["purity_value"])

                if not pure_sub_tbl.empty:
                    agg_purity = (
                        pure_sub_tbl.groupby(sample_col, as_index=False)[
                            "purity_value"
                        ]
                        .mean()
                        .rename(columns={"purity_value": "Pureté moyenne"})
                    )
                    agg_purity["Pureté moyenne"] = agg_purity["Pureté moyenne"].round(2)
                else:
                    agg_purity = pd.DataFrame(columns=[sample_col, "Pureté moyenne"])

                # Date fallback via pureté
                if "seizure_date" in pure_sub_tbl.columns:
                    pure_sub_tbl["seizure_date"] = _parse_dates_series(
                        pure_sub_tbl["seizure_date"]
                    )
                    last_date_pur = (
                        pure_sub_tbl[[sample_col, "seizure_date"]]
                        .dropna(subset=["seizure_date"])
                        .groupby(sample_col, as_index=False)["seizure_date"]
                        .max()
                        .rename(columns={"seizure_date": "Date"})
                    )
                    last_date_pur["Date"] = pd.to_datetime(
                        last_date_pur["Date"]
                    ).dt.strftime("%d/%m/%Y")
                else:
                    last_date_pur = pd.DataFrame(columns=[sample_col, "Date"])
            else:
                agg_purity = pd.DataFrame(columns=[sample_col, "Pureté moyenne"])
                last_date_pur = pd.DataFrame(columns=[sample_col, "Date"])

            # Poids net (somme par sample)
            weight_col = _find_weight_col(df_raw)
            if weight_col is None and "Poids net" in getattr(df_raw, "columns", []):
                weight_col = "Poids net"
            if weight_col and (weight_col in getattr(df_raw, "columns", [])):
                wsub = df_raw[[sample_col, weight_col]].copy()
                wsub[sample_col] = wsub[sample_col].astype(str)
                wsub[weight_col] = pd.to_numeric(wsub[weight_col], errors="coerce")
                agg_weight = (
                    wsub.groupby(sample_col, as_index=False)[weight_col]
                    .sum()
                    .rename(columns={weight_col: "Poids net"})
                )
                agg_weight["Poids net"] = agg_weight["Poids net"].round(2)
            else:
                agg_weight = pd.DataFrame(columns=[sample_col, "Poids net"])

            # Fusion
            tbl = base_samples.merge(agg_purity, on=sample_col, how="left")
            tbl = tbl.merge(
                last_date_raw if not last_date_raw.empty else last_date_pur,
                on=sample_col,
                how="left",
            )
            tbl = tbl.merge(agg_weight, on=sample_col, how="left")
            tbl = tbl.rename(columns={sample_col: "sample"})
            tbl = tbl[
                [
                    c
                    for c in ["sample", "Pureté moyenne", "Date", "Poids net"]
                    if c in tbl.columns
                ]
            ]

            st.dataframe(tbl, use_container_width=True, height=380)
            st.download_button(
                "⬇️ Exporter la liste (CSV)",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"cluster_{chosen_cluster}_samples_resume.csv",
                mime="text/csv",
                key="btn_dl_cluster_resume",
            )

    # ----- Pureté
    with tab_purity:
        st.markdown("**Boxplot de pureté pour le cluster sélectionné**")
        pure_long_all = (
            _melt_regressors(df_raw) if df_raw is not None else pd.DataFrame()
        )
        if pure_long_all.empty:
            st.info("Aucune colonne regressors_* détectée dans les données brutes.")
        else:
            pure_sub = (
                pure_long_all[
                    pure_long_all[sample_col].astype(str).isin(sample_list)
                ].copy()
                if (sample_col in pure_long_all.columns)
                else pure_long_all.copy()
            )
            if pure_sub.empty:
                st.info("Aucune donnée de pureté pour ces échantillons.")
            else:
                subs_avail = (
                    sorted(pure_sub["substance"].dropna().astype(str).unique())
                    if "substance" in pure_sub.columns
                    else []
                )
                chosen_sub_p = (
                    st.selectbox("Substance", subs_avail, key="cluster_detail_subsel")
                    if subs_avail
                    else None
                )
                if chosen_sub_p:
                    pure_sub = pure_sub[
                        pure_sub["substance"].astype(str) == chosen_sub_p
                    ]

                types_avail = (
                    sorted(pure_sub["purity_type_display"].dropna().unique().tolist())
                    if "purity_type_display" in pure_sub.columns
                    else []
                )
                chosen_type_p = (
                    st.selectbox(
                        "Type de pureté", types_avail, key="cluster_detail_typesel"
                    )
                    if types_avail
                    else None
                )

                date_col_present = "seizure_date" in pure_sub.columns
                if date_col_present:
                    pure_sub["seizure_date"] = _parse_dates_series(
                        pure_sub["seizure_date"]
                    )
                    min_d = pd.to_datetime(pure_sub["seizure_date"]).min()
                    max_d = pd.to_datetime(pure_sub["seizure_date"]).max()
                    d1p, d2p = st.date_input(
                        "Période (optionnel)",
                        value=(
                            (
                                min_d.date()
                                if pd.notna(min_d)
                                else datetime(2000, 1, 1).date()
                            ),
                            (
                                max_d.date()
                                if pd.notna(max_d)
                                else datetime.today().date()
                            ),
                        ),
                        key="cluster_detail_period",
                    )
                else:
                    d1p = d2p = None

                base_p = pure_sub.copy()
                if chosen_type_p:
                    base_p = base_p[base_p["purity_type_display"] == chosen_type_p]
                base_p = base_p.dropna(subset=["purity_value"])

                key_sample = (
                    sample_col
                    if (sample_col in base_p.columns)
                    else ("sample_name" if "sample_name" in base_p.columns else None)
                )
                if key_sample is None:
                    key_sample = "sample"
                    base_p[key_sample] = "S" + base_p.index.astype(str)

                if date_col_present and d1p and d2p:
                    base_p = base_p[
                        (base_p["seizure_date"] >= pd.to_datetime(d1p))
                        & (base_p["seizure_date"] <= pd.to_datetime(d2p))
                    ]

                if base_p.empty:
                    st.info("Aucune donnée avec ces filtres.")
                else:
                    agg_mode = st.radio(
                        "Agrégation",
                        ["Tous", "Année", "Mois", "Semaines"],
                        horizontal=True,
                        index=0,
                        key="cluster_detail_aggmode",
                    )

                    def label_bucket(dt: pd.Timestamp, mode: str) -> str:
                        if mode == "Année":
                            return dt.strftime("%Y")
                        if mode == "Mois":
                            return dt.strftime("%Y-%m")
                        if mode == "Semaines":
                            y, w, _ = dt.isocalendar()
                            return f"{int(y)}-W{int(w):02d}"
                        return "Tout"

                    if agg_mode == "Tous" or "seizure_date" not in base_p.columns:
                        base_p["bucket"] = "Tout"
                    else:
                        base_p = base_p.dropna(subset=["seizure_date"]).copy()
                        base_p["bucket"] = base_p["seizure_date"].apply(
                            lambda x: label_bucket(x, agg_mode)
                        )

                    grouped_purity = (
                        base_p.groupby(["bucket", key_sample], as_index=False)[
                            "purity_value"
                        ]
                        .mean()
                        .rename(columns={"purity_value": "purity_mean"})
                    )

                    # Ordre des buckets
                    if agg_mode == "Tous":
                        bucket_order = ["Tout"]
                    else:

                        def bucket_key(b):
                            if agg_mode == "Année":
                                return pd.Timestamp(int(b), 1, 1)
                            if agg_mode == "Mois":
                                y, m = b.split("-")
                                return pd.Timestamp(int(y), int(m), 1)
                            if agg_mode == "Semaines":
                                y, w = b.split("-W")
                                return pd.to_datetime(
                                    f"{y}-W{w}-1", format="%G-W%V-%u"
                                )
                            return pd.Timestamp(1900, 1, 1)

                        bucket_order = sorted(
                            grouped_purity["bucket"].unique(), key=bucket_key
                        )

                    st.altair_chart(
                        alt.Chart(grouped_purity)
                        .mark_boxplot(size=30)
                        .encode(
                            x=alt.X(
                                "bucket:N",
                                sort=bucket_order,
                                title=("Période" if agg_mode != "Tous" else ""),
                            ),
                            y=alt.Y(
                                "purity_mean:Q",
                                title=f"Pureté (moy./sample){' — '+chosen_type_p if chosen_type_p else ''}",
                            ),
                            tooltip=[
                                alt.Tooltip("bucket:N", title="Période"),
                                alt.Tooltip(
                                    "purity_mean:Q", title="Pureté (moy. sample)"
                                ),
                            ],
                        )
                        .properties(height=380),
                        use_container_width=True,
                    )

    # ----- Temporalité
    with tab_time:
        st.markdown("**Temporalité des échantillons du cluster**")
        if (
            (subset_raw.empty)
            or (not date_col_raw)
            or (date_col_raw not in subset_raw.columns)
        ):
            st.info("Aucune date disponible pour ces échantillons.")
        else:
            temp = subset_raw
            cols_ = [sample_col, date_col_raw] + (
                [namex_col_raw] if namex_col_raw else []
            )
            temp = temp[cols_].copy()
            temp[sample_col] = temp[sample_col].astype(str)
            temp[date_col_raw] = _parse_dates_series(temp[date_col_raw])

            temp_last = (
                temp.dropna(subset=[date_col_raw])
                .sort_values([sample_col, date_col_raw])
                .groupby(sample_col, as_index=False)
                .tail(1)
                .rename(columns={date_col_raw: "date"})
            )
            if temp_last.empty:
                st.info("Aucune date valide pour ces échantillons.")
            else:
                temp_last["date_day"] = pd.to_datetime(temp_last["date"]).dt.floor("D")
                if namex_col_raw and (namex_col_raw in temp_last.columns):
                    temp_last["_stack"] = (
                        temp_last.sort_values([namex_col_raw, "date_day", sample_col])
                        .groupby([namex_col_raw, "date_day"])
                        .cumcount()
                        + 1
                    )
                    color_enc = alt.Color(f"{namex_col_raw}:N", title="Substance")
                else:
                    temp_last["_stack"] = (
                        temp_last.sort_values(["date_day", sample_col])
                        .groupby(["date_day"])
                        .cumcount()
                        + 1
                    )
                    color_enc = alt.value("#2563EB")

                plot_df = temp_last.rename(columns={sample_col: "sample"})
                y_scale = alt.Scale(domain=[0, int(plot_df["_stack"].max()) + 1])

                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_point(size=90)
                    .encode(
                        x=alt.X("date_day:T", title="Date"),
                        y=alt.Y(
                            "_stack:Q",
                            axis=alt.Axis(title=None, ticks=False, labels=False),
                            scale=y_scale,
                        ),
                        color=color_enc,
                        tooltip=[
                            alt.Tooltip("sample:N", title="Sample"),
                            alt.Tooltip("date_day:T", title="Date", format="%d/%m/%Y"),
                            *(
                                [alt.Tooltip(f"{namex_col_raw}:N", title="Substance")]
                                if (namex_col_raw and namex_col_raw in plot_df.columns)
                                else []
                            ),
                        ],
                    )
                    .properties(height=180),
                    use_container_width=True,
                )

    # ----- Carte
    with tab_map:
        st.markdown("**Localisation des échantillons du cluster**")

        lat_col, lon_col = (
            _find_geo_cols(subset_raw) if not subset_raw.empty else (None, None)
        )

        if subset_raw.empty or not lat_col or not lon_col:
            st.info("Pas de colonnes lat/lon ou pas de données correspondantes.")
        else:
            gmap = subset_raw.dropna(subset=[lat_col, lon_col]).copy()
            gmap[lat_col] = pd.to_numeric(gmap[lat_col], errors="coerce")
            gmap[lon_col] = pd.to_numeric(gmap[lon_col], errors="coerce")
            gmap = gmap.dropna(subset=[lat_col, lon_col])

            if gmap.empty:
                st.info("Aucun point géolocalisé dans ce cluster.")
            else:
                # ⚙️ Choix du mode : spatial simple vs spatio-temporel
                mode = st.radio(
                    "Mode d’affichage",
                    ["Spatial uniquement", "Spatio-temporel"],
                    horizontal=True,
                    key="cluster_map_mode",
                )

                # ---------- Gestion de la date (si dispo) ----------
                has_date = bool(date_col_raw and date_col_raw in gmap.columns)
                if has_date:
                    gmap[date_col_raw] = _parse_dates_series(gmap[date_col_raw])
                    gmap["date_str"] = pd.to_datetime(
                        gmap[date_col_raw], errors="coerce"
                    ).dt.strftime("%d/%m/%Y")
                else:
                    gmap["date_str"] = ""

                # On garde toujours l'ordre temporel si la date existe
                if has_date:
                    gmap = gmap.sort_values(date_col_raw)
                    gmap["time_rank"] = (
                        gmap[date_col_raw]
                        .rank(method="first")
                        .astype(int)
                    )
                else:
                    gmap["time_rank"] = np.nan

                # ---------- Mode SPATIO-TEMPOREL ----------
                if (mode == "Spatio-temporel") and has_date:
                    # Slider "jusqu'à la date"
                    dmin = gmap[date_col_raw].min()
                    dmax = gmap[date_col_raw].max()
                    d_view = st.slider(
                        "Afficher les échantillons jusqu’à la date :",
                        min_value=dmin.date(),
                        max_value=dmax.date(),
                        value=dmax.date(),
                        key="cluster_map_date_max",
                    )
                    gmap = gmap[gmap[date_col_raw] <= pd.to_datetime(d_view)]

                    # Normalisation 0–1 pour la couleur temporelle
                    if len(gmap) > 1:
                        t_norm = (
                            (gmap[date_col_raw] - dmin)
                            / (dmax - dmin)
                        ).astype("float64")
                    else:
                        t_norm = pd.Series([0.5] * len(gmap), index=gmap.index)

                    def _time_to_rgba(v):
                        # v ∈ [0,1] => interpolation bleu → orange
                        v = max(0.0, min(1.0, float(v)))
                        r_old, g_old, b_old = 37, 99, 235    # bleu
                        r_new, g_new, b_new = 249, 115, 22   # orange
                        r = int(r_old + (r_new - r_old) * v)
                        g = int(g_old + (g_new - g_old) * v)
                        b = int(b_old + (b_new - b_old) * v)
                        return [r, g, b, 210]

                    gmap["_rgb"] = t_norm.apply(_time_to_rgba)
                    st.caption(
                        "Couleur = temps (bleu = plus ancien, orange = plus récent)."
                    )

                # ---------- Mode SPATIAL UNIQUEMENT (ou pas de date) ----------
                else:
                    # couleur fixe pour tous les points
                    gmap["_rgb"] = [[37, 99, 235, 210]] * len(gmap)
                    if mode == "Spatio-temporel" and not has_date:
                        st.info(
                            "Pas de colonne de date disponible pour ce cluster — "
                            "affichage spatio-temporel indisponible, passage en mode spatial."
                        )
                    else:
                        st.caption(
                            "Mode spatial simple : tous les points ont la même couleur."
                        )

                # ---------- Tooltip ----------
                tooltip_fields = []
                if sample_col in gmap.columns:
                    tooltip_fields.append(("Sample", sample_col))
                if "date_str" in gmap.columns and has_date:
                    tooltip_fields.append(("Date", "date_str"))
                if "time_rank" in gmap.columns and has_date:
                    tooltip_fields.append(("Ordre temporel", "time_rank"))

                tooltip_html = (
                    "<br/>".join(
                        [f"<b>{lbl}:</b> {{{col}}}" for lbl, col in tooltip_fields]
                    )
                    if tooltip_fields
                    else ""
                )

                # ---------- Vue automatique ----------
                try:
                    from pydeck.data_utils import viewport as pdk_viewport

                    pts = gmap[[lon_col, lat_col]].dropna().values
                    vp = pdk_viewport.compute_view(pts)
                    view_state = pdk.ViewState(
                        float(vp["latitude"]),
                        float(vp["longitude"]),
                        float(vp["zoom"]),
                        min_zoom=2,
                        max_zoom=16,
                    )
                except Exception:
                    view_state = pdk.ViewState(46.5, 2.5, 6, min_zoom=2, max_zoom=16)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=gmap,
                    get_position=[lon_col, lat_col],
                    get_radius=400,
                    radius_min_pixels=6,
                    radius_max_pixels=80,
                    get_fill_color="_rgb",
                    pickable=True,
                )
                deck = deck_with_fallback(
                    [layer],
                    view_state,
                    {"html": tooltip_html} if tooltip_html else None,
                )
                st.pydeck_chart(deck, use_container_width=True, height=520)


# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
with st.sidebar:
    st.title("Menu")
    page = st.radio(
        "Aller à…",
        [
            "Accueil",
            "Données",
            "Statistiques",
            "Carte",
            "Profilage",
            "Valises marocaines",
        ],
    )
    st.markdown("---")


# ============================================================
# Pages
# ============================================================

# ------------------------ Accueil ---------------------------
if page == "Accueil":
    st.title("Dashboard Projet Argos-Signature")
    st.write("Analyse des données du projet Argos_Signature")
    try:
        dt_modif = datetime.fromtimestamp(os.path.getmtime(__file__)).strftime(
            "%d/%m/%Y à %H:%M:%S"
        )
        st.markdown(f"🕓 **Dernière mise à jour de l'application :** {dt_modif}")
    except Exception as e:
        st.warning(f"Impossible de récupérer la date de mise à jour ({e})")

# ------------------------ Données ---------------------------
elif page == "Données":
    st.title("📈 Données")
    st.caption("Importe un Excel/CSV, choisis la feuille, explore et filtre.")

    uploaded = st.file_uploader(
        "Dépose un fichier Excel (.xlsx/.xls) ou CSV",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False,
    )

    if uploaded:
        ext = _ext(uploaded.name)
        try:
            if ext == "csv":
                df = _read_csv(uploaded)
            elif ext in ("xlsx", "xls"):
                df_tmp, sheets = _read_excel(uploaded)
                if len(sheets) > 1:
                    uploaded.seek(0)
                    chosen = st.selectbox("Feuille Excel", sheets, index=0)
                    df, _ = _read_excel(uploaded, sheet_name=chosen)
                else:
                    df = df_tmp
            else:
                st.error("Format non supporté.")
                st.stop()
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            st.stop()

        st.session_state["df"] = df

        st.subheader("Aperçu")
        st.dataframe(df, use_container_width=True, height=420)

        st.markdown("---")
        st.subheader("Filtrer")
        query = st.text_input(
            "Contient (toutes colonnes) :", placeholder="ex: client, REF-2025, etc."
        )
        if query and query.strip():
            q = query.strip()
            filtered = df[
                df.apply(
                    lambda r: r.astype(str)
                    .str.contains(q, case=False, na=False)
                    .any(),
                    axis=1,
                )
            ]
        else:
            filtered = df
        st.caption(f"Résultats: {len(filtered)} ligne(s)")

        st.dataframe(filtered, use_container_width=True, height=300)

        st.subheader("Exporter")
        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="donnees_filtrees.csv",
            mime="text/csv",
        )
    else:
        st.info("Glisse-dépose un fichier Excel/CSV ci-dessus pour commencer.")

# ---------------------- Statistiques ------------------------
elif page == "Statistiques":
    st.title("📊 Statistiques")
    df = st.session_state.get("df")
    if df is None:
        st.warning(
            "⚠️ Aucun fichier en mémoire — importe d’abord un fichier dans l’onglet Données."
        )
        st.stop()

    cols = df.columns.tolist()
    sample_col = _find_col(["sample_name"], cols)
    seizure_col = _find_col(["seizure", "seizure_id", "seizureid", "id_seizure"], cols)
    namex_col = _find_col(["name_x"], cols)
    date_col = _find_col(["seizure_date"], cols)

    # --- KPIs
    st.subheader("Décompte")
    kpi1 = df[sample_col].dropna().nunique() if sample_col in df.columns else 0
    kpi2 = (
        df[seizure_col].dropna().nunique()
        if seizure_col and seizure_col in df.columns
        else 0
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">Échantillons</div>'
            f'<div class="kpi-value">{kpi1:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-label">Affaires</div>'
            f'<div class="kpi-value">{kpi2:,}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Substances
    st.subheader("Substances analysées")
    if namex_col and sample_col:

        # Appliquer regroupements de substances
        df[namex_col] = df[namex_col].replace(SUBSTANCE_GROUPS)

        # 1 ligne par échantillon
        base_unique = df.dropna(subset=[sample_col]).drop_duplicates(
            subset=[sample_col]
        )

        counts = (
            base_unique.dropna(subset=[namex_col])
            .groupby(namex_col)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        if counts.empty:
            st.info("Aucune substance à afficher.")
        else:
            counts[namex_col] = counts[namex_col].astype(str)
            counts["%"] = (counts["count"] / counts["count"].sum() * 100).round(2)

            substances = counts[namex_col].tolist()

            # Couleurs fixes par substance
            if "fixed_colors" not in st.session_state:
                st.session_state["fixed_colors"] = {}
            fixed = st.session_state["fixed_colors"]

            color_index = len(fixed)
            for s in substances:
                if s not in fixed:
                    fixed[s] = PALETTE[color_index % len(PALETTE)]
                    color_index += 1

            cmap = fixed

            measure = st.radio(
                "Mesure",
                ["Nombre", "Pourcentage (%)"],
                horizontal=True,
                index=0,
                key="subs_measure",
            )

            with st.expander("Tableau des statistiques par substance"):
                st.dataframe(
                    counts.rename(columns={"count": "Nombre", "%": "Pourcentage (%)"}),
                    use_container_width=True,
                    height=300,
                )

            total_samples = counts["count"].sum()
            if measure == "Nombre":
                x_title = f"Nombre d'échantillons (n = {total_samples:,})"
            else:
                x_title = f"Proportion (%) (n = {total_samples:,})"

            y_field = "count" if measure == "Nombre" else "%"

            bars = (
                alt.Chart(counts)
                .mark_bar(size=18)
                .encode(
                    y=alt.Y(
                        f"{namex_col}:N",
                        sort="-x",
                        title="Substances",
                        axis=alt.Axis(labelLimit=300),
                    ),
                    x=alt.X(
                        f"{y_field}:Q",
                        title=x_title,
                    ),
                    color=alt.Color(
                        f"{namex_col}:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=substances,
                            range=[cmap[s] for s in substances],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{namex_col}:N", title="Substance"),
                        alt.Tooltip("count:Q", title="Nombre"),
                        alt.Tooltip("%:Q", title="Proportion (%)"),
                    ],
                )
            )

            labels = (
                alt.Chart(counts)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=5,
                    fontSize=12,
                    color="#111827",
                )
                .encode(
                    y=alt.Y(f"{namex_col}:N", sort="-x"),
                    x=alt.X(f"{y_field}:Q"),
                    text=alt.Text(
                        f"{y_field}:Q",
                        format=",.1f" if measure != "Nombre" else ",d",
                    ),
                )
            )

            chart = (bars + labels).properties(height=40 * len(counts))

            st.altair_chart(chart, use_container_width=True)

    else:
        st.caption("Section ignorée (colonnes manquantes).")

    # --- Séries temporelles
    st.subheader("📅 Nombre de spécimens par période")
    if not date_col:
        st.error("Colonne `seizure_date` introuvable.")
        st.stop()

    unit = st.radio("Unité :", ["Echantillons", "Affaires"], horizontal=True)
    period = st.radio("Période :", ["Mois", "Semaines"], horizontal=True)
    display_mode = st.radio(
        "Affichage :",
        ["Total (toutes substances)", "Par substance (empilé)"],
        horizontal=True,
    )
    compare_both = st.checkbox(
        "Afficher Echantillons et Affaires côte à côte (si disponibles)",
        value=False,
    )

    u = unit.lower()
    if "echantillon" in u:
        key_col = sample_col
    else:
        key_col = seizure_col

    if not key_col:
        st.error("Colonne d’unité introuvable (sample/seizure).")
        st.stop()

    selected_namex = []
    if namex_col:
        selected_namex = st.multiselect(
            "Filtrer certaines substances (`name_x`)",
            options=sorted(df[namex_col].dropna().astype(str).unique()),
            default=[],
        )

    base = (
        df[[key_col, date_col] + ([namex_col] if namex_col else [])]
        .dropna(subset=[key_col, date_col])
        .copy()
    )
    base[date_col] = _parse_dates_series(base[date_col])
    base = (
        base.dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[key_col], keep="first")
    )

    if namex_col and selected_namex:
        base = base[base[namex_col].astype(str).isin(selected_namex)]

    base_both = None
    if compare_both and sample_col and seizure_col:
        base_samples = (
            df[[sample_col, date_col] + ([namex_col] if namex_col else [])]
            .dropna(subset=[sample_col, date_col])
            .copy()
        )
        base_samples[date_col] = _parse_dates_series(base_samples[date_col])
        base_samples = (
            base_samples.dropna(subset=[date_col])
            .sort_values(date_col)
            .drop_duplicates(subset=[sample_col], keep="first")
        )
        base_samples["unite"] = "Echantillons"

        base_seizures = (
            df[[seizure_col, date_col] + ([namex_col] if namex_col else [])]
            .dropna(subset=[seizure_col, date_col])
            .copy()
        )
        base_seizures[date_col] = _parse_dates_series(base_seizures[date_col])
        base_seizures = (
            base_seizures.dropna(subset=[date_col])
            .sort_values(date_col)
            .drop_duplicates(subset=[seizure_col], keep="first")
        )
        base_seizures["unite"] = "Affaires"

        if namex_col and selected_namex:
            base_samples = base_samples[
                base_samples[namex_col].astype(str).isin(selected_namex)
            ]
            base_seizures = base_seizures[
                base_seizures[namex_col].astype(str).isin(selected_namex)
            ]

        base_both = pd.concat([base_samples, base_seizures], ignore_index=True)

    if period == "Mois":
        base["period"] = base[date_col].dt.to_period("M").dt.to_timestamp()
        fmt, nice = "%b %Y", "month"
    else:
        base["period"] = base[date_col].dt.to_period("W").apply(lambda p: p.start_time)
        fmt, nice = "W%V %Y", "week"

    if base_both is not None:
        if period == "Mois":
            base_both["period"] = (
                base_both[date_col].dt.to_period("M").dt.to_timestamp()
            )
        else:
            base_both["period"] = (
                base_both[date_col].dt.to_period("W").apply(lambda p: p.start_time)
            )

    if display_mode.startswith("Total") or not namex_col:
        grouped = (
            base.groupby("period")
            .size()
            .reset_index(name="count")
            .sort_values("period")
        )
        color_enc = None
    else:
        grouped = (
            base.groupby(["period", namex_col])
            .size()
            .reset_index(name="count")
            .sort_values(["period", namex_col])
        )
        order = (
            grouped.groupby(namex_col)["count"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        cmap = _ensure_color_map(
            order, st.session_state.get("substance_colors", {}), PALETTE
        )
        st.session_state["substance_colors"] = cmap
        color_enc = alt.Color(
            f"{namex_col}:N",
            sort=order,
            legend=alt.Legend(title=namex_col),
            scale=alt.Scale(domain=order, range=[cmap[s] for s in order]),
        )

    with st.expander("Voir le tableau"):
        st.dataframe(grouped, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger l’agrégat (CSV)",
        data=grouped.to_csv(index=False).encode("utf-8"),
        file_name=f"compte_par_{period.lower().replace('è','e')}.csv",
        mime="text/csv",
    )

    n = grouped["period"].nunique()
    bar_size = (
        48 if n <= 15 else 32 if n <= 30 else 20 if n <= 60 else 12 if n <= 120 else 8
    )

    if period == "Mois":
        mois_fr = [
            "janv.",
            "févr.",
            "mars",
            "avr.",
            "mai",
            "juin",
            "juil.",
            "août",
            "sept.",
            "oct.",
            "nov.",
            "déc.",
        ]
        grouped = grouped.copy()
        grouped["period_label"] = (
            grouped["period"].dt.month.apply(lambda m: mois_fr[m - 1])
            + " "
            + grouped["period"].dt.year.astype(str)
        )
        period_order = grouped.sort_values("period")["period_label"].tolist()
        x_enc = alt.X(
            "period_label:N",
            sort=period_order,
            axis=alt.Axis(title=None, labelAngle=0, labelPadding=8),
        )
        tooltip_period = alt.Tooltip("period_label:N", title="Période")
        x_scale = alt.Scale()
    else:
        tick_count = min(n, 8)
        x_axis = alt.Axis(
            format=fmt, labelAngle=0, labelPadding=8, tickCount=tick_count
        )
        x_scale = alt.Scale(nice=nice)
        x_enc = alt.X("period:T", axis=x_axis, scale=x_scale, title=None)
        tooltip_period = alt.Tooltip("period:T", title="Période", format=fmt)

    if display_mode.startswith("Total") or color_enc is None:
        base_chart = alt.Chart(grouped).encode(
            x=x_enc,
            y=alt.Y("count:Q", title=None),
            tooltip=[
                tooltip_period,
                alt.Tooltip("count:Q", title="Unités uniques"),
            ],
        )
        chart = base_chart.mark_bar(
            size=bar_size,
            color="#2563EB",
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        if n <= 60:
            chart = chart + base_chart.mark_text(
                dy=-6, fontSize=13, color="#111827"
            ).encode(text="count:Q")
    else:
        chart = (
            alt.Chart(grouped)
            .mark_bar(size=bar_size, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=x_enc,
                y=alt.Y("count:Q", stack="zero", title=None),
                color=color_enc,
                tooltip=[
                    tooltip_period,
                    alt.Tooltip("count:Q", title="Unités uniques"),
                    alt.Tooltip(f"{namex_col}:N", title="Substance"),
                ],
            )
        )

    st.altair_chart(chart.properties(height=420), use_container_width=True)

    if compare_both and base_both is not None and display_mode.startswith("Total"):
        grouped_both = (
            base_both.groupby(["period", "unite"])
            .size()
            .reset_index(name="count")
            .sort_values(["period", "unite"])
        )

        with st.expander("Voir le tableau comparatif Echantillons / Affaires"):
            st.dataframe(grouped_both, use_container_width=True)

        n_both = grouped_both["period"].nunique()
        bar_size_both = (
            40 if n_both <= 15 else 28 if n_both <= 30 else 18 if n_both <= 60 else 10
        )

        if period == "Mois":
            grouped_both = grouped_both.copy()
            grouped_both["period_label"] = (
                grouped_both["period"].dt.month.apply(lambda m: mois_fr[m - 1])
                + " "
                + grouped_both["period"].dt.year.astype(str)
            )
            period_order_both = grouped_both.sort_values("period")[
                "period_label"
            ].tolist()
            x_enc_both = alt.X(
                "period_label:N",
                sort=period_order_both,
                axis=alt.Axis(title=None, labelAngle=0, labelPadding=8),
            )
            tooltip_period_both = alt.Tooltip("period_label:N", title="Période")
        else:
            x_enc_both = alt.X("period:T", axis=x_axis, scale=x_scale, title=None)
            tooltip_period_both = alt.Tooltip("period:T", title="Période", format=fmt)

        chart_both = (
            alt.Chart(grouped_both)
            .mark_bar(size=bar_size_both)
            .encode(
                x=x_enc_both,
                y=alt.Y("count:Q", title="Unités uniques"),
                color=alt.Color(
                    "unite:N",
                    title="Unité",
                    scale=alt.Scale(
                        domain=["Echantillons", "Affaires"],
                        range=["#2563EB", "#F97316"],
                    ),
                ),
                xOffset="unite:N",
                tooltip=[
                    tooltip_period_both,
                    alt.Tooltip("unite:N", title="Unité"),
                    alt.Tooltip("count:Q", title="Unités uniques"),
                ],
            )
            .properties(
                height=420,
                title="Comparaison Echantillons vs Affaires par période",
            )
        )

        st.altair_chart(chart_both, use_container_width=True)

    st.markdown("---")

    # --- Pureté : boxplots & médianes (multi-substances + multi-types) ---
    st.subheader("🧪 Pureté — Boxplots par période")

    pure_long = _melt_regressors(df)
    if pure_long.empty:
        st.error("Aucune colonne `regressors_*` détectée (type + numeric).")
        st.stop()
    if "substance" not in pure_long.columns or pure_long["substance"].dropna().empty:
        st.error("Colonne 'substance' / 'name_x' introuvable.")
        st.stop()

    all_substances = sorted(pure_long["substance"].dropna().astype(str).unique())

    freq = (
        pure_long["substance"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "substance", "substance": "n"})
    )
    default_subs = [s for s in freq["substance"].tolist() if s in all_substances][:3]

    selected_substances = st.multiselect(
        "Substances à quantifier",
        options=all_substances,
        default=default_subs or all_substances[:1],
        key="purity_substances",
    )
    if not selected_substances:
        st.info("Sélectionne au moins une substance.")
        st.stop()

    pure_long = pure_long[pure_long["substance"].isin(selected_substances)].copy()

    types_avail = sorted(pure_long["purity_type_display"].dropna().unique().tolist())
    chosen_types = st.multiselect(
        "Quantification(s) (type de pureté)",
        options=types_avail,
        default=types_avail[:2] if len(types_avail) >= 2 else types_avail,
        key="purity_types",
    )
    if not chosen_types:
        st.info("Sélectionne au moins un type de pureté.")
        st.stop()

    date_col_present = "seizure_date" in pure_long.columns
    if date_col_present:
        pure_long["seizure_date"] = _parse_dates_series(pure_long.get("seizure_date"))
        min_d = pd.to_datetime(pure_long["seizure_date"].min())
        max_d = pd.to_datetime(pure_long["seizure_date"].max())
        d1, d2 = st.date_input(
            "Période (optionnel)",
            value=(
                (min_d.date() if pd.notna(min_d) else datetime(2000, 1, 1).date()),
                (max_d.date() if pd.notna(max_d) else datetime.today().date()),
            ),
            key="purity_date_range",
        )
    else:
        d1 = d2 = None

    def _bucket_label(dt: pd.Timestamp, mode: str) -> str:
        if mode == "Année":
            return dt.strftime("%Y")
        if mode == "Mois":
            return dt.strftime("%Y-%m")
        if mode == "Semaines":
            y, w, _ = dt.isocalendar()
            return f"{int(y)}-W{int(w):02d}"
        return "Tout"

    agg = st.radio(
        "Agrégation temporelle",
        ["Tous", "Année", "Mois", "Semaines"],
        horizontal=True,
        key="purity_agg",
    )

    base_p = pure_long[pure_long["purity_type_display"].isin(chosen_types)].copy()
    base_p = base_p[
        [
            "sample_name",
            "seizure_date",
            "purity_value",
            "substance",
            "purity_type_display",
        ]
    ].dropna(subset=["purity_value", "sample_name"])

    if date_col_present:
        base_p["seizure_date"] = _parse_dates_series(base_p["seizure_date"])
        if d1 and d2:
            base_p = base_p[
                (base_p["seizure_date"] >= pd.to_datetime(d1))
                & (base_p["seizure_date"] <= pd.to_datetime(d2))
            ]

    if base_p.empty:
        st.info("Aucune donnée pour ces filtres.")
        st.stop()

    if agg == "Tous" or "seizure_date" not in base_p.columns:
        base_p["bucket"] = "Tout"
    else:
        base_p = base_p.dropna(subset=["seizure_date"]).copy()
        base_p["bucket"] = base_p["seizure_date"].apply(lambda x: _bucket_label(x, agg))

    if agg == "Tous":
        bucket_order = ["Tout"]
    else:

        def bucket_key(b):
            if agg == "Année":
                return pd.Timestamp(int(b), 1, 1)
            if agg == "Mois":
                y, m = b.split("-")
                return pd.Timestamp(int(y), int(m), 1)
            if agg == "Semaines":
                y, w = b.split("-W")
                return pd.to_datetime(f"{y}-W{w}-1", format="%G-W%V-%u")
            return pd.Timestamp(1900, 1, 1)

        bucket_order = sorted(base_p["bucket"].unique(), key=bucket_key)

    base_width = max(600, 80 * len(bucket_order))

    grouped_purity = (
        base_p.groupby(
            ["bucket", "sample_name", "substance", "purity_type_display"],
            as_index=False,
        )["purity_value"]
        .mean()
        .rename(columns={"purity_value": "purity_mean"})
    )

    fixed = st.session_state.get("fixed_colors", {})
    color_index = len(fixed)
    for s in selected_substances:
        if s not in fixed:
            fixed[s] = PALETTE[color_index % len(PALETTE)]
            color_index += 1
    st.session_state["fixed_colors"] = fixed
    cmap = fixed

    color_scale = alt.Scale(
        domain=selected_substances,
        range=[cmap[s] for s in selected_substances],
    )

    ymode = st.radio(
        "Échelle Y", ["Auto", "0–100"], index=0, horizontal=True, key="purity_ybox"
    )

    if ymode == "0–100":
        y_box = alt.Y(
            "purity_mean:Q",
            title="Pureté (moyenne par échantillon) (%)",
            scale=alt.Scale(domain=[0, 100]),
        )
    else:
        y_box = alt.Y(
            "purity_mean:Q",
            title="Pureté (moyenne par échantillon) (%)",
        )

    chart_box_base = (
        alt.Chart(grouped_purity)
        .mark_boxplot(size=18)
        .encode(
            x=alt.X(
                "bucket:N",
                sort=bucket_order,
                title=("Période" if agg != "Tous" else ""),
            ),
            xOffset=alt.XOffset("substance:N"),
            y=y_box,
            color=alt.Color(
                "substance:N",
                title="Substance",
                scale=color_scale,
            ),
            tooltip=[
                alt.Tooltip("bucket:N", title="Période"),
                alt.Tooltip("substance:N", title="Substance"),
                alt.Tooltip("purity_type_display:N", title="Type de pureté"),
                alt.Tooltip("purity_mean:Q", title="Pureté (moy./échantillon)"),
            ],
        )
        .properties(height=380, width=base_width)
    )

    facet_title = ", ".join(selected_substances)

    chart_box = (
        chart_box_base.facet(
            column=alt.Column(
                "purity_type_display:N",
                title=facet_title,
                sort=chosen_types,
            )
        ).resolve_scale(y="independent")
    )

    st.altair_chart(chart_box, use_container_width=True)

    st.markdown("#### Pureté médiane et variabilité par type de pureté")

    show_band = st.checkbox(
        "Afficher la bande ±1 écart-type autour de la médiane",
        value=True,
        key="purity_show_sd_band",
    )

    ts_stats = (
        grouped_purity.groupby(
            ["bucket", "substance", "purity_type_display"]
        )["purity_mean"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            sd="std",
        )
        .reset_index()
    )

    ts_stats["rsd"] = (ts_stats["sd"] / ts_stats["mean"]) * 100
    ts_stats["rsd"] = (
        ts_stats["rsd"].replace([np.inf, -np.inf], np.nan).clip(lower=0)
    )

    ts_stats["lower"] = (ts_stats["median"] - ts_stats["sd"]).clip(lower=0)
    if ymode == "0–100":
        ts_stats["upper"] = (ts_stats["median"] + ts_stats["sd"]).clip(upper=100)
    else:
        ts_stats["upper"] = ts_stats["median"] + ts_stats["sd"]

    ts_stats["bucket"] = pd.Categorical(
        ts_stats["bucket"], categories=bucket_order, ordered=True
    )
    ts_stats = ts_stats.sort_values("bucket")

    base_ts = alt.Chart(ts_stats).encode(
        x=alt.X("bucket:N", sort=bucket_order, title="Période"),
        color=alt.Color(
            "substance:N",
            title="Substance",
            scale=color_scale,
        ),
    )

    if ymode == "0–100":
        y_band = alt.Y(
            "lower:Q",
            title="Pureté médiane (%)",
            scale=alt.Scale(domain=[0, 100]),
        )
        y_line = alt.Y(
            "median:Q",
            title="Pureté médiane (%)",
            scale=alt.Scale(domain=[0, 100]),
        )
    else:
        y_band = alt.Y(
            "lower:Q",
            title="Pureté médiane (%)",
        )
        y_line = alt.Y(
            "median:Q",
            title="Pureté médiane (%)",
        )

    if show_band:
        band = base_ts.mark_area(opacity=0.18).encode(
            y=y_band,
            y2="upper:Q",
        )
    else:
        band = None

    line = base_ts.mark_line(point=True, strokeWidth=2).encode(
        y=y_line,
        tooltip=[
            alt.Tooltip("bucket:N", title="Période"),
            alt.Tooltip("substance:N", title="Substance"),
            alt.Tooltip("purity_type_display:N", title="Type de pureté"),
            alt.Tooltip("median:Q", title="Médiane (%)"),
            alt.Tooltip("sd:Q", title="Écart-type"),
            alt.Tooltip("rsd:Q", title="RSD (%)"),
            alt.Tooltip("n:Q", title="n échantillons"),
        ],
    )

    layer_ts = line if not show_band else band + line
    layer_ts_base = layer_ts.properties(height=260, width=base_width)

    chart_ts = (
        layer_ts_base.facet(
            column=alt.Column(
                "purity_type_display:N",
                title=facet_title,
                sort=chosen_types,
            )
        ).resolve_scale(y="independent")
    )

    st.altair_chart(chart_ts, use_container_width=True)

    recap = (
        grouped_purity.groupby(
            ["bucket", "substance", "purity_type_display"]
        )["purity_mean"]
        .agg(
            n_samples="count",
            mean="mean",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values(
            ["bucket", "substance", "purity_type_display"],
            key=lambda s: (
                [bucket_order.index(x) if x in bucket_order else -1 for x in s]
                if s.name == "bucket"
                else s
            ),
        )
    )

    with st.expander(
        "Tableau des statistiques par période, substance et type de pureté"
    ):
        st.dataframe(recap, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger les données agrégées (CSV)",
        data=grouped_purity.to_csv(index=False).encode("utf-8"),
        file_name=f"purete_{agg}.csv",
        mime="text/csv",
    )

# ------------------------- Carte ----------------------------
elif page == "Carte":
    st.title("🗺️ Carte des échantillons")
    df = st.session_state.get("df")
    if df is None:
        st.warning(
            "⚠️ Aucun fichier en mémoire — importe d’abord un fichier dans l’onglet Données."
        )
        st.stop()

    cols = df.columns.tolist()
    sample_col = _find_col(["sample_name"], cols)
    namex_col = _find_col(["name_x"], cols)
    date_col = _find_col(["seizure_date"], cols)
    lat_col, lon_col = _find_geo_cols(df)

    if not lat_col or not lon_col:
        st.error("Colonnes latitude/longitude introuvables.")
        st.stop()

    st.caption(
        f"Colonnes utilisées : latitude = **{lat_col}**, longitude = **{lon_col}**"
    )

    base_cols = [lat_col, lon_col] + [c for c in (sample_col, namex_col, date_col) if c]
    gdf = df[base_cols].dropna(subset=[lat_col, lon_col]).copy()
    gdf[lat_col] = pd.to_numeric(gdf[lat_col], errors="coerce")
    gdf[lon_col] = pd.to_numeric(gdf[lon_col], errors="coerce")
    gdf = gdf.dropna(subset=[lat_col, lon_col])

    if date_col:
        gdf[date_col] = _parse_dates_series(gdf[date_col])
        min_d = pd.to_datetime(gdf[date_col].min())
        max_d = pd.to_datetime(gdf[date_col].max())
        d1, d2 = st.date_input(
            "Période (optionnel)",
            value=(
                (min_d.date() if pd.notna(min_d) else datetime(2000, 1, 1).date()),
                (max_d.date() if pd.notna(max_d) else datetime.today().date()),
            ),
        )
        gdf = gdf[
            (gdf[date_col] >= pd.to_datetime(d1))
            & (gdf[date_col] <= pd.to_datetime(d2))
        ]

    if sample_col:
        gdf = gdf.sort_values(date_col if date_col else lat_col).drop_duplicates(
            subset=[sample_col], keep="first"
        )

    if namex_col:
        options = sorted(gdf[namex_col].dropna().astype(str).unique())
        chosen = st.multiselect(
            "Filtrer par substance (`name_x`)", options=options, default=[]
        )
        if chosen:
            gdf = gdf[gdf[namex_col].astype(str).isin(chosen)]

    if gdf.empty:
        st.info("Aucun point à afficher avec ces filtres.")
        st.stop()

    cluster_overlay = st.session_state.get("cluster_overlay")
    use_clusters = False
    filter_range = None
    mode = "Points"

    if cluster_overlay and "assignments" in cluster_overlay and sample_col:
        st.markdown("### Clusters")
        use_clusters = st.checkbox(
            "Filtrer par les plus grands clusters (couleur par cluster)",
            value=False,
            help="Nécessite d’avoir généré un dendrogramme et des clusters dans l’onglet Profilage.",
        )
        if use_clusters:
            assign_df = cluster_overlay["assignments"]
            gdf = gdf.merge(assign_df, on=sample_col, how="inner")

            n_top_default = int(cluster_overlay.get("n_top", 10))
            saved_top_ids = cluster_overlay.get("top_ids")

            n_top_map = st.number_input(
                "Nombre de plus grands clusters à afficher",
                min_value=1,
                max_value=int(max(1, gdf["cluster"].nunique())),
                value=n_top_default,
                step=1,
                key="n_top_clusters_map",
            )

            if (
                saved_top_ids
                and len(saved_top_ids) >= 1
                and len(saved_top_ids) == n_top_default
            ):
                top_ids = set(saved_top_ids[: int(n_top_map)])
            else:
                top_ids = (
                    gdf["cluster"]
                    .value_counts()
                    .sort_values(ascending=False)
                    .head(int(n_top_map))
                    .index
                )
                top_ids = set(top_ids)

            gdf = gdf[gdf["cluster"].isin(top_ids)]
            if gdf.empty:
                st.warning(
                    f"Aucun point parmi les {int(n_top_map)} plus grands clusters n’est visible avec les filtres actuels."
                )

    if use_clusters and ("cluster" in gdf.columns):
        uniq_clusters = sorted(gdf["cluster"].unique().tolist())
        cmap = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(uniq_clusters)}
        gdf["_rgb"] = gdf["cluster"].map(cmap).apply(lambda c: _hex_to_rgba(c, 200))

        dc = _find_col(["seizure_date", "date"], gdf.columns) or date_col
        if dc and dc in gdf.columns:
            gdf[dc] = _parse_dates_series(gdf[dc])
            gdf["_ts"] = pd.to_datetime(gdf[dc], errors="coerce").view("int64") // 10**9
            if gdf["_ts"].notna().any():
                ts_min, ts_max = int(gdf["_ts"].min()), int(gdf["_ts"].max())
                dmin_ui, dmax_ui = (
                    pd.to_datetime(ts_min, unit="s").date(),
                    pd.to_datetime(ts_max, unit="s").date(),
                )

                st.markdown("#### Période (glisser pour voir apparaître/disparaître)")
                mode_temps = st.radio(
                    "Mode",
                    ["Fenêtre mobile", "Cumul jusqu’à…"],
                    horizontal=True,
                    key="cluster_time_mode",
                )
                if mode_temps == "Fenêtre mobile":
                    d1_sel, d2_sel = st.slider(
                        "Fenêtre temporelle",
                        min_value=dmin_ui,
                        max_value=dmax_ui,
                        value=(dmin_ui, dmax_ui),
                        format="DD/MM/YYYY",
                        key="cluster_time_window",
                    )
                    filter_range = [
                        int(pd.Timestamp(d1_sel).timestamp()),
                        int(pd.Timestamp(d2_sel).timestamp()),
                    ]
                else:
                    d_sel = st.slider(
                        "Cumul (≤ date)",
                        min_value=dmin_ui,
                        max_value=dmax_ui,
                        value=dmax_ui,
                        format="DD/MM/YYYY",
                        key="cluster_time_cum",
                    )
                    filter_range = [ts_min, int(pd.Timestamp(d_sel).timestamp())]

                if filter_range:
                    st.caption(
                        f"Période affichée : "
                        f"{pd.to_datetime(filter_range[0], unit='s').date():%d/%m/%Y} → "
                        f"{pd.to_datetime(filter_range[1], unit='s').date():%d/%m/%Y}"
                    )
    else:
        mode = st.radio("Mode d’affichage", ["Points", "Heatmap"], horizontal=True)

        if namex_col:
            subs = sorted(gdf[namex_col].dropna().astype(str).unique())

            fixed = st.session_state.get("fixed_colors", {})
            color_index = len(fixed)
            for s in subs:
                if s not in fixed:
                    fixed[s] = PALETTE[color_index % len(PALETTE)]
                    color_index += 1

            st.session_state["fixed_colors"] = fixed
            cmap_subs = fixed

            gdf["_rgb"] = (
                gdf[namex_col]
                .astype(str)
                .map(cmap_subs)
                .where(lambda s: s.notna(), "#2563EB")
                .apply(lambda c: _hex_to_rgba(c, 180))
            )
        else:
            gdf["_rgb"] = [[37, 99, 235, 180]] * len(gdf)

    gdf_view = gdf.copy()
    if (filter_range is not None) and ("_ts" in gdf_view.columns):
        lo, hi = filter_range
        gdf_view = gdf_view[(gdf_view["_ts"] >= lo) & (gdf_view["_ts"] <= hi)]
    if gdf_view.empty:
        st.info("Aucun point à afficher pour la période sélectionnée.")
        st.stop()

    view_state = pdk.ViewState(
        latitude=46.5,
        longitude=2.5,
        zoom=6,
        min_zoom=2,
        max_zoom=16,
        bearing=0,
        pitch=0,
    )

    tooltip_fields = []
    if sample_col:
        tooltip_fields.append(("Sample", sample_col))
    if namex_col:
        tooltip_fields.append(("Substance", namex_col))
    if date_col and (date_col in gdf_view.columns):
        gdf_view["date_str"] = pd.to_datetime(
            gdf_view[date_col], errors="coerce"
        ).dt.strftime("%d/%m/%Y")
        tooltip_fields.append(("Date", "date_str"))
    if use_clusters and ("cluster" in gdf_view.columns):
        tooltip_fields.append(("Cluster", "cluster"))
    tooltip_html = (
        "<br/>".join([f"<b>{lbl}:</b> {{{col}}}" for lbl, col in tooltip_fields])
        if tooltip_fields
        else ""
    )

    layer = pdk.Layer(
        "ScatterplotLayer" if mode == "Points" else "HeatmapLayer",
        data=gdf_view,
        get_position=[lon_col, lat_col],
        **(
            {
                "get_radius": 400,
                "radius_min_pixels": 6,
                "radius_max_pixels": 80,
                "get_fill_color": "_rgb",
                "pickable": True,
            }
            if mode == "Points"
            else {"radiusPixels": 40}
        ),
    )
    st.pydeck_chart(
        deck_with_fallback(
            [layer], view_state, {"html": tooltip_html} if tooltip_html else None
        ),
        use_container_width=True,
        height=800,
    )
    st.caption(f"Points visibles dans la période sélectionnée : {len(gdf_view):,}")

    st.download_button(
        "⬇️ Télécharger les points affichés (CSV)",
        data=gdf_view.drop(columns=["_rgb"], errors="ignore")
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="points_carte.csv",
        mime="text/csv",
    )

# ------------------------ Profilage -------------------------
elif page == "Profilage":
    st.title("Profilage")

    df = st.session_state.get("df")
    if df is None:
        st.warning(
            "⚠️ Aucun fichier en mémoire — importe d’abord un fichier dans l’onglet Données."
        )
        st.stop()

    # --- Spectres présents ? ---
    spec_cols, wls = _get_spectrum_cols(df)
    if not spec_cols:
        st.error("Aucune colonne spectrale détectée (attendu: 'spectrum_XXXX').")
        st.stop()

    # ============================
    # 1) Sidebar : pipeline
    # ============================
    with st.sidebar:
        st.markdown("### ⚙️ Prétraitements (ordre appliqué)")
        pipeline = st.multiselect(
            "Pipeline",
            ["Découpe", "SG dérivée 2", "SNV"],
            default=["SG dérivée 2", "Découpe", "SNV"],
            help="Les étapes seront exécutées dans l'ordre sélectionné.",
        )

        wl_min, wl_max = float(wls.min()), float(wls.max())
        default_lo, default_hi = max(int(wl_min), 1100), min(int(wl_max), 1645)
        if default_lo >= default_hi:
            default_lo, default_hi = int(wl_min), int(wl_max)

        cut_range = st.slider(
            "Plage (longueurs d’onde)",
            int(wl_min),
            int(wl_max),
            (default_lo, default_hi),
        )

        c1, c2 = st.columns(2)
        with c1:
            sg_window = st.number_input(
                "Fenêtre SG (impair)", min_value=3, max_value=301, step=2, value=5
            )
        with c2:
            sg_poly = st.number_input(
                "Ordre poly SG", min_value=2, max_value=5, step=1, value=2
            )

        st.caption("L'application se met à jour automatiquement selon vos choix.")

    if pipeline and (apply_savgol is None or apply_snv is None):
        st.error("Le module 'preprocessing.py' n'a pas pu être importé.")
        st.stop()

    # ============================
    # 2) Exécution pipeline
    # ============================
        # --- Exécution pipeline ---
    working = df.copy()

    if pipeline and (apply_savgol is None or apply_snv is None):
        st.error("Le module 'preprocessing.py' n'a pas pu être importé.")
        st.stop()

    if pipeline:
        # on réutilise la fonction générique (comportement identique)
        working = run_spectral_pipeline(
            working,
            pipeline=pipeline,
            cut_range=cut_range,
            sg_window=sg_window,
            sg_poly=sg_poly,
            prefix="spectrum_",
        )

    out = working

    # ============================
    # 3) Visualisation des spectres
    # ============================
    st.subheader("Visualiser les spectres (après prétraitements)")
    spec_cols_final, _ = _get_spectrum_cols(out)
    if len(spec_cols_final) < 2:
        st.error("Pas assez de colonnes spectrales pour tracer les spectres.")
    else:
        sample_col = _find_col(["sample_name", "Sample"], out.columns)

        # --- Cas avec sample_name ---
        if sample_col:
            all_samples = sorted(out[sample_col].dropna().astype(str).unique().tolist())

            s_query = st.text_input(
                "Rechercher un sample_name", placeholder="ex: ABC123"
            )
            if s_query:
                filt = [s for s in all_samples if s_query.lower() in s.lower()]
            else:
                filt = all_samples

            max_curves = st.slider(
                "Nombre max. de courbes à superposer", 1, 30, 10, 1
            )
            selected = st.multiselect(
                "Sélectionner un ou plusieurs samples", options=filt, default=filt[:1]
            )

            avg_before_plot = st.checkbox(
                "Moyenner les réplicas par sample avant tracé", value=False
            )

            if not selected:
                st.info("Choisis au moins un sample pour afficher les courbes.")
            else:
                df_plot = out[out[sample_col].astype(str).isin(selected)].copy()

                if avg_before_plot:
                    df_plot = (
                        df_plot[[sample_col] + spec_cols_final]
                        .groupby(sample_col, as_index=False)
                        .mean(numeric_only=True)
                    )
                    df_plot["_series"] = df_plot[sample_col].astype(str)
                else:
                    df_plot["_rep"] = df_plot.groupby(sample_col).cumcount() + 1
                    df_plot["_series"] = (
                        df_plot[sample_col].astype(str)
                        + " · r"
                        + df_plot["_rep"].astype(str)
                    )

                if len(df_plot) > max_curves:
                    st.warning(
                        f"{len(df_plot)} courbes → affichage limité aux {max_curves} premières."
                    )
                    df_plot = df_plot.head(max_curves)

                long = df_plot[[sample_col, "_series"] + spec_cols_final].melt(
                    id_vars=[sample_col, "_series"],
                    var_name="wl",
                    value_name="intensity",
                )
                long["wl"] = (
                    long["wl"]
                    .astype(str)
                    .str.replace("spectrum_", "", regex=False)
                    .astype(float)
                )

                st.altair_chart(
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("wl:Q", title="Longueur d’onde", sort="ascending"),
                        y=alt.Y("intensity:Q", title="Intensité"),
                        color=alt.Color("_series:N", title="Sample · réplica"),
                        detail=alt.Detail("_series:N"),
                        tooltip=[
                            alt.Tooltip(f"{sample_col}:N", title="Sample"),
                            alt.Tooltip("_series:N", title="Réplica"),
                            alt.Tooltip("wl:Q", title="λ"),
                            alt.Tooltip("intensity:Q", title="Intensité"),
                        ],
                    )
                    .properties(height=420),
                    use_container_width=True,
                )

                with st.expander("⬇️ Export des données affichées"):
                    export_df = long.pivot(
                        index=[sample_col, "_series"], columns="wl", values="intensity"
                    ).reset_index()
                    st.download_button(
                        "Télécharger les spectres affichés (CSV)",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name="spectres_affiches.csv",
                        mime="text/csv",
                    )

        # --- Cas sans sample_name : sélection par index ---
        else:
            st.info("Colonne 'sample_name' introuvable — sélection par index de ligne.")
            idxs = st.multiselect(
                "Index de lignes à tracer",
                options=out.index.tolist(),
                default=out.index.tolist()[:1],
            )
            if idxs:
                df_plot = out.loc[idxs, spec_cols_final].copy()
                df_plot["label"] = [str(i) for i in idxs]
                long = df_plot.reset_index(drop=True).melt(
                    id_vars=["label"], var_name="wl", value_name="intensity"
                )
                long["wl"] = (
                    long["wl"]
                    .astype(str)
                    .str.replace("spectrum_", "", regex=False)
                    .astype(float)
                )
                st.altair_chart(
                    alt.Chart(long)
                    .mark_line()
                    .encode(
                        x=alt.X("wl:Q", title="Longueur d’onde"),
                        y=alt.Y("intensity:Q", title="Intensité"),
                        color=alt.Color("label:N", title="Ligne"),
                        tooltip=[
                            alt.Tooltip("label:N", title="Ligne"),
                            alt.Tooltip("wl:Q", title="λ"),
                            alt.Tooltip("intensity:Q", title="Intensité"),
                        ],
                    )
                    .properties(height=420),
                    use_container_width=True,
                )

    # Export de tous les prétraitements
    with st.expander("Exporter le résultat complet des prétraitements (CSV)"):
        st.download_button(
            "⬇️ Télécharger le résultat complet",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="pretraitements_resultat.csv",
            mime="text/csv",
        )

    # ============================
    # 4) Clustering & dendrogramme
    # ============================
    st.markdown("---")
    st.subheader("Dendrogramme des spectres (distance euclidienne)")

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        import plotly.figure_factory as ff
    except Exception as e:
        st.error(f"Cette section requiert SciPy et Plotly. Import impossible : {e}")
        st.stop()

    namex_col = _find_col(
        ["name_x", "substance", "substance_name", "drug"], out.columns
    )
    sample_col = _find_col(["sample_name", "Sample"], out.columns)
    spec_cols_final, _ = _get_spectrum_cols(out)
    date_col_out = _find_col(["seizure_date", "date"], out.columns)

    if not namex_col:
        st.info("Colonne 'name_x' introuvable : impossible de choisir une substance.")
    elif len(spec_cols_final) < 2:
        st.info("Pas assez de colonnes spectrales pour calculer des distances.")
    else:
        # --- Choix de la substance et des paramètres de clustering ---
        col1, col2 = st.columns(2)
        with col1:
            subs_options = sorted(out[namex_col].dropna().astype(str).unique())
            target_variants = {
                "résine thc",
                "resine thc",
                "résine de thc",
                "resine de thc",
                "thc resin",
            }
            default_sub_idx = 0
            for i, s in enumerate(subs_options):
                if s.strip().casefold() in target_variants:
                    default_sub_idx = i
                    break
            chosen_substance = st.selectbox(
                "Substance (name_x)",
                subs_options,
                index=default_sub_idx,
                key="dendro_substance",
            )

        with col2:
            methods = ["ward", "average", "complete", "single"]
            link_method = st.selectbox(
                "Méthode de liaison",
                methods,
                index=methods.index("complete"),
                key="dendro_linkage",
            )

        colA, colB = st.columns(2)
        with colA:
            avg_rep = st.checkbox(
                "Moyenner les réplicas par sample", value=bool(sample_col)
            )
        with colB:
            color_threshold = st.number_input(
                "Seuil de distance (0 = auto)", min_value=0.0, step=0.1, value=0.0
            )

        # --- Filtre temporel optionnel ---
        if date_col_out:
            tmp = out[out[namex_col].astype(str) == str(chosen_substance)].copy()
            if tmp.empty:
                tmp = out.copy()
            tmp[date_col_out] = _parse_dates_series(tmp[date_col_out])
            min_d = pd.to_datetime(tmp[date_col_out].min())
            max_d = pd.to_datetime(tmp[date_col_out].max())
            d1, d2 = st.date_input(
                "Période (filtre appliqué au dendrogramme)",
                value=(
                    (min_d.date() if pd.notna(min_d) else datetime(2000, 1, 1).date()),
                    (max_d.date() if pd.notna(max_d) else datetime.today().date()),
                ),
            )
        else:
            d1 = d2 = None

        # --- Sous-ensemble pour la substance choisie ---
        df_sub = out[out[namex_col].astype(str) == str(chosen_substance)].copy()
        if date_col_out and d1 and d2:
            df_sub[date_col_out] = _parse_dates_series(df_sub[date_col_out])
            df_sub = df_sub[
                (df_sub[date_col_out] >= pd.to_datetime(d1))
                & (df_sub[date_col_out] <= pd.to_datetime(d2))
            ]

        if df_sub.empty:
            st.warning("Aucun échantillon pour cette substance sur la période choisie.")
            st.stop()

        M = df_sub[spec_cols_final].apply(pd.to_numeric, errors="coerce")
        mask_valid = M.notna().all(axis=1)
        if mask_valid.sum() < 2:
            st.warning("Moins de 2 spectres valides — pas de dendrogramme.")
            st.stop()

        M = M.loc[mask_valid]
        df_sub_valid = df_sub.loc[mask_valid]

        # Moyenne des réplicas si demandé
        if avg_rep and sample_col and sample_col in df_sub_valid.columns:
            M[sample_col] = df_sub_valid[sample_col].astype(str).values
            M = M.groupby(sample_col, as_index=True).mean()
            labels, X = M.index.astype(str).tolist(), M.values
        else:
            labels = (
                df_sub_valid[sample_col].astype(str).tolist()
                if (sample_col and sample_col in df_sub_valid.columns)
                else df_sub_valid.index.astype(str).tolist()
            )
            X = M.values

        def _linkagefun(x: np.ndarray):
            """Wrapper sur linkage pour gérer le choix de la méthode."""
            if link_method == "ward":
                return linkage(x, method="ward", metric="euclidean")
            return linkage(x, method=link_method, metric="euclidean")

        # --- Dendrogramme Plotly ---
        fig = ff.create_dendrogram(
            X,
            orientation="bottom",
            labels=labels,
            linkagefun=_linkagefun,
            color_threshold=(None if color_threshold == 0 else float(color_threshold)),
        )
        fig.update_layout(
            title=f"Dendrogramme — {chosen_substance} (méthode: {link_method})",
            xaxis_title="Échantillons",
            yaxis_title="Distance",
            height=620,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )

        # ============================
        # 5) Attribution de clusters
        # ============================
        st.markdown("##### Attribution de clusters")

        Z = _linkagefun(X)

        cluster_mode = st.radio(
            "Découpe",
            ["Par nombre de clusters (k)", "Par seuil de distance"],
            horizontal=True,
            index=1,
        )
        if cluster_mode == "Par nombre de clusters (k)":
            k = st.slider(
                "k (clusters)",
                min_value=2,
                max_value=max(2, min(20, len(labels))),
                value=4,
                step=1,
            )
            clusters = fcluster(Z, t=k, criterion="maxclust")
        else:
            thr = st.number_input(
                "Seuil de distance",
                min_value=0.0,
                value=float(color_threshold or 0.0),
                step=0.1,
            )
            clusters = fcluster(
                Z, t=(thr if thr > 0 else Z[:, 2].mean()), criterion="distance"
            )

        cluster_df = pd.DataFrame({"label": labels, "cluster": clusters.astype(int)})
        sizes = (
            cluster_df["cluster"]
            .value_counts()
            .rename("taille")
            .reset_index()
            .rename(columns={"index": "cluster"})
        )
        cluster_df_full = cluster_df.merge(sizes, on="cluster", how="left")

        tri_mode = st.selectbox(
            "Trier le tableau par…",
            [
                "Taille du cluster (décroissante)",
                "ID de cluster (ascendant)",
                "Label (ascendant)",
            ],
        )
        if tri_mode.startswith("Taille"):
            cluster_df_sorted = cluster_df_full.sort_values(
                by=["taille", "cluster", "label"], ascending=[False, True, True]
            )
        elif tri_mode.startswith("ID de cluster"):
            cluster_df_sorted = cluster_df_full.sort_values(
                by=["cluster", "label"], ascending=[True, True]
            )
        else:
            cluster_df_sorted = cluster_df_full.sort_values(
                by=["label", "cluster"], ascending=[True, True]
            )

        n_clusters = cluster_df["cluster"].nunique()
        st.markdown(
            f"**Nombre de clusters : {n_clusters}**  &nbsp;|&nbsp; "
            f"**Total échantillons : {len(cluster_df)}**"
        )
        st.dataframe(
            cluster_df_sorted.rename(
                columns={
                    "cluster": "Cluster",
                    "taille": "Taille",
                    "label": "Échantillon",
                }
            ),
            use_container_width=True,
            height=320,
        )

        # ============================
        # 6) Recherche rapide par sample
        # ============================
        with st.expander(
            "🔎 Rechercher un sample par nom (auto-complétion 'contient')",
            expanded=False,
        ):
            q = st.text_input(
                "Tape quelques lettres du sample_name…",
                placeholder="ex: ABC, 2305, …",
                key="cluster_sample_query",
            )

            labels_unique = sorted(cluster_df["label"].astype(str).unique().tolist())
            suggestions = (
                [s for s in labels_unique if q.strip().lower() in s.lower()][:100]
                if q and q.strip()
                else []
            )

            if suggestions:
                pick = st.selectbox(
                    "Résultats :",
                    options=suggestions,
                    index=0,
                    key="cluster_sample_pick",
                )
                cl_ids = sorted(
                    cluster_df.loc[cluster_df["label"] == pick, "cluster"]
                    .unique()
                    .tolist()
                )
                st.caption(
                    f"Ce sample appartient au(x) cluster(s) : "
                    f"{', '.join(map(str, cl_ids))}"
                )

                goto = st.selectbox(
                    "Aller au cluster…", options=cl_ids, index=0, key="cluster_goto_id"
                )
                if st.button("➡️ Afficher ce cluster"):
                    st.session_state["cluster_choice_dropdown"] = int(goto)
                    st.session_state["cluster_choice_text"] = str(goto)
                    st.session_state["cluster_focus_id"] = int(goto)
                    st.success(f"Cluster {int(goto)} sélectionné.")
            else:
                if q and q.strip():
                    st.info("Aucun sample ne correspond à cette recherche.")
                else:
                    st.caption("Astuce : commence à taper pour voir les suggestions.")

        # ============================
        # 7) Sélection d’un cluster
        # ============================
        uniq_ids = sorted(cluster_df["cluster"].unique().tolist())
        csel1, csel2 = st.columns([1, 1])

        with csel1:
            cluster_choice_dropdown = st.selectbox(
                "Choisir un cluster (liste)",
                options=uniq_ids,
                index=0,
                key="cluster_choice_dropdown",
                help="Sélection via la liste déroulante.",
            )

        with csel2:
            cluster_choice_text = st.text_input(
                "…ou saisir l’ID du cluster",
                value=str(cluster_choice_dropdown),
                key="cluster_choice_text",
                help=(
                    "Entre un entier (ex. 3). "
                    "Si l’ID n’existe pas, la valeur de la liste est conservée."
                ),
            )
            selected_cluster_id = cluster_choice_dropdown
            try:
                typed = int(cluster_choice_text.strip())
                if typed in uniq_ids:
                    selected_cluster_id = typed
                else:
                    st.info(
                        f"ID {typed} introuvable parmi les clusters existants : "
                        f"{uniq_ids[:10]}{' …' if len(uniq_ids) > 10 else ''}"
                    )
            except Exception:
                pass

            st.session_state["cluster_focus_id"] = selected_cluster_id

            focus_preview = cluster_df_full.loc[
                cluster_df_full["cluster"] == selected_cluster_id
            ]
            with st.expander(f"🔎 Détail rapide — Cluster {selected_cluster_id}"):
                st.dataframe(
                    focus_preview.rename(
                        columns={"label": "Échantillon", "taille": "Taille"}
                    ),
                    use_container_width=True,
                    height=260,
                )

        # ============================
        # 8) Top X plus grands clusters
        # ============================
        n_clusters_total = cluster_df["cluster"].nunique()
        n_top = st.number_input(
            "Nombre de plus grands clusters à afficher",
            min_value=1,
            max_value=int(max(1, n_clusters_total)),
            value=min(10, int(n_clusters_total)),
            step=1,
            key="n_top_clusters",
        )

        top_clusters = (
            cluster_df_full.groupby(["cluster"], as_index=False)
            .agg(
                taille=("label", "count"),
                exemples=("label", lambda s: ", ".join(s.head(3))),
            )
            .sort_values("taille", ascending=False)
            .head(int(n_top))
        )

        with st.expander(f"🧩 Top {int(n_top)} des clusters les plus grands"):
            st.dataframe(
                top_clusters.rename(
                    columns={
                        "cluster": "Cluster",
                        "taille": "Taille",
                        "exemples": "Exemples",
                    }
                ),
                use_container_width=True,
                height=300,
            )

        # ============================
        # 9) Sauvegarde pour Carte & Détail de cluster
        # ============================
        if sample_col:
            cluster_assign = (
                cluster_df[["label", "cluster"]]
                .rename(columns={"label": sample_col})
                .copy()
            )
            top_ids = (
                cluster_df["cluster"]
                .value_counts()
                .sort_values(ascending=False)
                .head(int(n_top))
                .index.tolist()
            )
            st.session_state["cluster_overlay"] = {
                "sample_col": sample_col,
                "assignments": cluster_assign,
                "top_ids": top_ids,
                "substance": chosen_substance,
                "method": link_method,
                "n_top": int(n_top),
            }

        st.download_button(
            "⬇️ Exporter attribution (CSV)",
            data=cluster_df.to_csv(index=False).encode("utf-8"),
            file_name=f"clusters_{chosen_substance}_{link_method}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Exporter tableau trié (CSV)",
            data=cluster_df_sorted.to_csv(index=False).encode("utf-8"),
            file_name=f"clusters_{chosen_substance}_{link_method}_tableau_trie.csv",
            mime="text/csv",
        )

        # Payload pour la section Détail d’un cluster + rendu
        st.session_state["cluster_detail"] = {
            "cluster_df": cluster_df,
            "cluster_df_full": cluster_df_full,
            "sample_col": sample_col,
            "out": out,
            "df_raw": st.session_state.get("df"),
            "date_col_raw": date_col_out,
            "namex_col_raw": namex_col,
        }
        render_cluster_detail_section()

# --------------------- Valises marocaines -------------------
elif page == "Valises marocaines":
    # ============================================================
    # Valises marocaines — page isolée (import, prétraitements, PCA/t-SNE, variabilité)
    # ============================================================
    st.title("🧳 Valises marocaines")
    st.caption(
        "Importer un ou plusieurs Excel, appliquer des prétraitements (depuis le menu), "
        "faire une PCA/t-SNE et analyser la variabilité intra/inter. Cette page est isolée des autres."
    )

    # ------------------------------------------------------------
    # 1) IMPORT MULTI-EXCEL / MULTI-FEUILLES
    # ------------------------------------------------------------
    st.markdown("### Importer des fichiers Excel")
    vm_uploaded_files = st.file_uploader(
        "Importer un ou plusieurs fichiers Excel (.xlsx / .xls)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="vm_upload",
    )

    def _vm_read_selected_sheets(xls_file, selected_sheets):
        """Concatène les feuilles sélectionnées d'un ExcelFile en un DataFrame."""
        frames = []
        for sh in selected_sheets:
            try:
                frames.append(pd.read_excel(xls_file, sheet_name=sh))
            except Exception as e:
                st.warning(
                    f"Feuille '{sh}' illisible ({getattr(xls_file, 'io', 'fichier')}) : {e}"
                )
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )

    vm_imported_frames = []
    if vm_uploaded_files:
        st.info(
            "Sélectionne les feuilles à utiliser pour chaque fichier (par défaut : toutes)."
        )
        for i, f in enumerate(vm_uploaded_files):
            try:
                xls = pd.ExcelFile(f)
            except Exception as e:
                st.error(f"Impossible d’ouvrir '{f.name}' : {e}")
                continue

            sheets = xls.sheet_names
            st.caption(f"**{f.name}** — Feuilles : {', '.join(sheets)}")
            vm_selected = st.multiselect(
                f"Feuilles à importer pour {f.name}",
                options=sheets,
                default=sheets,
                key=f"vm_sheets_{i}",
            )
            if not vm_selected:
                st.warning(f"Aucune feuille sélectionnée pour {f.name} — ignoré.")
                continue

            try:
                f.seek(0)  # important avant une seconde lecture
                xls2 = pd.ExcelFile(f)
                vm_imported_frames.append(_vm_read_selected_sheets(xls2, vm_selected))
            except Exception as e:
                st.error(f"Lecture de '{f.name}' impossible : {e}")

    if vm_imported_frames:
        vm_df = pd.concat(vm_imported_frames, ignore_index=True, sort=False)
        st.session_state["valises_df"] = vm_df

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Lignes importées</div>'
                f'<div class="kpi-value">{len(vm_df):,}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Colonnes</div>'
                f'<div class="kpi-value">{len(vm_df.columns):,}</div></div>',
                unsafe_allow_html=True,
            )

        with st.expander("Aperçu des données importées"):
            st.dataframe(vm_df, use_container_width=True, height=360)
    else:
        vm_df = st.session_state.get("valises_df")
        if vm_df is None or vm_df.empty:
            st.info("Importe au moins un fichier Excel pour commencer.")
            st.stop()

    # ------------------------------------------------------------
    # 2) PRÉTRAITEMENTS (SIDEBAR)
    # ------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Prétraitements (Valises marocaines)")
        st.caption("Les étapes sont exécutées dans l'ordre sélectionné.")
        vm_pipeline = st.multiselect(
            "Pipeline",
            ["Découpe", "SG dérivée 2", "SNV"],
            default=["SG dérivée 2", "Découpe", "SNV"],
            help=(
                "Découpe = plage λ ; SG dérivée 2 = Savitzky–Golay dérivée 2 ; "
                "SNV = Standard Normal Variate."
            ),
            key="vm_pipeline",
        )

        vm_spec_cols, vm_wls = _get_spectrum_cols(vm_df)
        if vm_spec_cols:
            wl_min, wl_max = float(vm_wls.min()), float(vm_wls.max())
            lo, hi = max(int(wl_min), 1100), min(int(wl_max), 1645)
            if lo >= hi:
                lo, hi = int(wl_min), int(wl_max)

            st.session_state.setdefault("vm_cut_range", (lo, hi))
            st.slider(
                "Plage (longueurs d’onde)",
                int(wl_min),
                int(wl_max),
                st.session_state["vm_cut_range"],
                key="vm_cut_range",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Fenêtre SG (impair)", 3, 301, 5, 2, key="vm_sg_window"
                )
            with c2:
                st.number_input("Ordre poly SG", 2, 5, 2, 1, key="vm_sg_poly")
        else:
            st.caption("Aucune colonne spectrale détectée (préfixe 'spectrum_').")

        vm_out = vm_df.copy()

    # --- Application pipeline Valises ---
    if vm_pipeline:
        vm_spec_cols_check, _ = _get_spectrum_cols(vm_out)
        if not vm_spec_cols_check:
            st.warning("Prétraitements ignorés : aucune colonne spectrale détectée.")
        else:
            if ("SG dérivée 2" in vm_pipeline or "SNV" in vm_pipeline) and (
                apply_savgol is None or apply_snv is None
            ):
                st.error(
                    "Module 'preprocessing.py' introuvable — SG/SNV indisponibles."
                )
            else:
                wl_lo, wl_hi = st.session_state["vm_cut_range"]
                vm_out = run_spectral_pipeline(
                    vm_out,
                    pipeline=vm_pipeline,
                    cut_range=(wl_lo, wl_hi),
                    sg_window=int(st.session_state["vm_sg_window"]),
                    sg_poly=int(st.session_state["vm_sg_poly"]),
                    prefix="spectrum_",
                )


    st.session_state["valises_df_processed"] = vm_out

    st.markdown("---")
    st.subheader("Aperçu du résultat (après prétraitements)")
    st.dataframe(vm_out, use_container_width=True, height=380)
    st.download_button(
        "⬇️ Télécharger le résultat prétraité (CSV)",
        vm_out.to_csv(index=False).encode("utf-8"),
        "valises_pretraite.csv",
        "text/csv",
        key="vm_dl_pretraite",
    )

    # ------------------------------------------------------------
    # 3) ANALYSE (PCA ou t-SNE) — ISOLÉE À CETTE PAGE
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("Réduction de dimension")

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except Exception as e:
        st.error(f"scikit-learn requis (import échoué : {e})")
        st.stop()

    vm_spec_cols_pca, vm_wls_pca = _get_spectrum_cols(vm_out)
    if len(vm_spec_cols_pca) < 2:
        st.info("Analyse indisponible : moins de 2 colonnes spectrales détectées.")
        st.stop()

    vm_sample_col = _find_col(["sample_name", "Sample"], vm_out.columns)
    vm_namex_col = _find_col(
        ["name_x", "substance", "substance_name", "drug"], vm_out.columns
    )
    vm_date_col = _find_col(["seizure_date", "date"], vm_out.columns)
    vm_style_attrs = ["Localisation", "Lot", "Savonnette", "Logo", "sample_name"]

    vm_method = st.radio("Méthode", ["PCA", "t-SNE"], horizontal=True, key="vm_method")

    c1, c2, c3 = st.columns(3)
    with c1:
        if vm_method == "PCA":
            vm_n_comp = st.slider(
                "Composantes (PCA)",
                2,
                min(10, len(vm_spec_cols_pca)),
                5,
                1,
                key="vm_n_comp",
            )
        else:
            vm_n_comp_tsne = st.selectbox(
                "Dimensions (t-SNE)", [2, 3], 0, key="vm_tsne_dim"
            )
    with c2:
        vm_avg_reps = st.checkbox(
            "Moyenner par sample", value=bool(vm_sample_col), key="vm_avg_reps"
        )
    with c3:
        vm_point_size = st.slider(
            "Taille des points", 40, 300, 160, 10, key="vm_point_size"
        )

    if vm_method == "t-SNE":
        t1, t2, t3 = st.columns(3)
        with t1:
            vm_tsne_perp = st.slider("Perplexity", 5, 100, 30, 1, key="vm_tsne_perp")
        with t2:
            vm_tsne_lr = st.slider("Learning rate", 10, 2000, 200, 10, key="vm_tsne_lr")
        with t3:
            vm_tsne_iter = st.slider(
                "Itérations", 250, 5000, 1000, 50, key="vm_tsne_iter"
            )

    # --- Construction table de travail ---
    vm_work = vm_out.copy()
    if "sample_name" not in vm_work.columns:
        if vm_sample_col and vm_sample_col in vm_work.columns:
            vm_work["sample_name"] = vm_work[vm_sample_col].astype(str)
        else:
            vm_work["sample_name"] = "S" + vm_work.index.astype(str)

    if vm_avg_reps:
        vm_X_mean = vm_work.groupby("sample_name", as_index=False)[
            vm_spec_cols_pca
        ].mean(numeric_only=True)
        attrs_present = [
            c for c in vm_style_attrs if c in vm_work.columns and c != "sample_name"
        ]

        def _first_non_null(s: pd.Series):
            s = s.dropna()
            return s.iloc[0] if not s.empty else None

        if attrs_present:
            vm_A = vm_work.groupby("sample_name", as_index=False)[attrs_present].agg(
                _first_non_null
            )
            vm_work2 = vm_X_mean.merge(vm_A, on="sample_name", how="left")
        else:
            vm_work2 = vm_X_mean.copy()
    else:
        keep_attrs = [
            c for c in vm_style_attrs if c in vm_work.columns and c != "sample_name"
        ]
        vm_work2 = pd.concat(
            [
                vm_work[["sample_name"] + keep_attrs].reset_index(drop=True),
                vm_work[vm_spec_cols_pca].reset_index(drop=True),
            ],
            axis=1,
        ).loc[:, lambda df: ~df.columns.duplicated()]

    if (
        vm_namex_col
        and vm_namex_col in vm_out.columns
        and vm_namex_col not in vm_work2.columns
    ):
        vm_work2[vm_namex_col] = (
            vm_out[vm_namex_col]
            if not vm_avg_reps
            else vm_out[["sample_name", vm_namex_col]]
            .drop_duplicates("sample_name")[vm_namex_col]
            .values
        )

    if (
        vm_date_col
        and vm_date_col in vm_out.columns
        and vm_date_col not in vm_work2.columns
    ):
        if vm_avg_reps:
            meta_dt = vm_out[["sample_name", vm_date_col]].copy()
            meta_dt[vm_date_col] = _parse_dates_series(meta_dt[vm_date_col])
            meta_dt = (
                meta_dt.dropna(subset=[vm_date_col])
                .sort_values(["sample_name", vm_date_col])
                .groupby("sample_name", as_index=False)
                .tail(1)
            )
            vm_work2 = vm_work2.merge(meta_dt, on="sample_name", how="left")
        else:
            vm_work2[vm_date_col] = vm_out[vm_date_col]

    # --- Nettoyage & centrage ---
    X = vm_work2[vm_spec_cols_pca].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    if mask.sum() < 3:
        st.warning(
            "Analyse indisponible : moins de 3 lignes complètes après nettoyage."
        )
        st.stop()

    X = X.loc[mask]
    vm_work2 = vm_work2.loc[mask].reset_index(drop=True)
    Xc = X.values - X.values.mean(axis=0, keepdims=True)  # centrage simple

    # --- PCA ou t-SNE ---
    if vm_method == "PCA":
        vm_pca = PCA(n_components=vm_n_comp, random_state=0)
        vm_scores = vm_pca.fit_transform(Xc)
        vm_expl = vm_pca.explained_variance_ratio_
        comp_names = [f"PC{i}" for i in range(1, vm_n_comp + 1)]
    else:
        vm_tsne = TSNE(
            n_components=vm_n_comp_tsne,
            perplexity=vm_tsne_perp,
            learning_rate=vm_tsne_lr,
            n_iter=vm_tsne_iter,
            init="pca",
            random_state=0,
            verbose=0,
        )
        vm_scores = vm_tsne.fit_transform(Xc)
        vm_expl = None
        comp_names = [f"tSNE{i}" for i in range(1, vm_n_comp_tsne + 1)]

    # --- Scores + métadonnées ---
    attach_cols = list(
        dict.fromkeys(
            [
                c
                for c in (vm_style_attrs + [vm_namex_col, vm_date_col])
                if c and c in vm_work2.columns
            ]
        )
    )
    vm_scores_df = pd.concat(
        [
            vm_work2[attach_cols].reset_index(drop=True),
            pd.DataFrame(vm_scores, columns=comp_names),
        ],
        axis=1,
    ).loc[:, lambda df: ~df.columns.duplicated()]

    st.markdown("**Projection**")
    a1, a2 = st.columns(2)
    with a1:
        vm_pc_x = st.selectbox("Axe X", comp_names, 0, key="vm_rd_x")
    with a2:
        vm_pc_y = st.selectbox(
            "Axe Y", comp_names, 1 if len(comp_names) > 1 else 0, key="vm_rd_y"
        )

    present_attrs = [
        c
        for c in ["Localisation", "Lot", "Savonnette", "Logo", "sample_name"]
        if c in vm_scores_df.columns
    ]

    def _is_num(s: pd.Series) -> bool:
        try:
            return pd.api.types.is_numeric_dtype(s)
        except Exception:
            return False

    ccol, cshape = st.columns(2)
    with ccol:
        vm_color_attr = st.selectbox(
            "Couleur par…",
            ["(aucune)"] + present_attrs,
            (
                (present_attrs.index("Localisation") + 1)
                if "Localisation" in present_attrs
                else 0
            ),
            key="vm_rd_color_attr",
        )
    with cshape:
        cat_opts = [c for c in present_attrs if not _is_num(vm_scores_df[c])]
        vm_shape_attr = st.selectbox(
            "Forme par…", ["(aucune)"] + cat_opts, 0, key="vm_rd_shape_attr"
        )

    tips = []
    if "sample_name" in vm_scores_df.columns:
        tips.append(alt.Tooltip("sample_name:N", title="Sample"))
    if vm_namex_col and vm_namex_col in vm_scores_df.columns:
        tips.append(alt.Tooltip(f"{vm_namex_col}:N", title="Substance"))
    if vm_date_col and vm_date_col in vm_scores_df.columns:
        try:
            vm_scores_df[vm_date_col] = _parse_dates_series(vm_scores_df[vm_date_col])
        except Exception:
            pass
        tips.append(alt.Tooltip(f"{vm_date_col}:T", title="Date", format="%d/%m/%Y"))
    tips += [alt.Tooltip(f"{vm_pc_x}:Q"), alt.Tooltip(f"{vm_pc_y}:Q")]

    color_enc = (
        alt.value("#2563EB")
        if vm_color_attr == "(aucune)"
        else alt.Color(
            f"{vm_color_attr}:{'Q' if _is_num(vm_scores_df[vm_color_attr]) else 'N'}",
            title=vm_color_attr,
        )
    )
    shape_enc = (
        alt.value("circle")
        if vm_shape_attr == "(aucune)"
        else alt.Shape(f"{vm_shape_attr}:N", title=vm_shape_attr)
    )

    def _axis_title(ax: str) -> str:
        if vm_method == "PCA":
            idx = int(ax.replace("PC", "")) - 1
            return f"{ax} ({vm_expl[idx]*100:.1f}%)"
        return ax

    st.altair_chart(
        alt.Chart(vm_scores_df)
        .mark_point(size=vm_point_size, opacity=0.9)
        .encode(
            x=alt.X(f"{vm_pc_x}:Q", title=_axis_title(vm_pc_x)),
            y=alt.Y(f"{vm_pc_y}:Q", title=_axis_title(vm_pc_y)),
            color=color_enc,
            shape=shape_enc,
            tooltip=tips,
        )
        .properties(height=900),
        use_container_width=True,
    )

    with st.expander("⬇️ Exporter"):
        st.download_button(
            "Scores (CSV)",
            vm_scores_df.to_csv(index=False).encode("utf-8"),
            "pca_scores.csv" if vm_method == "PCA" else "tsne_scores.csv",
            "text/csv",
            key="vm_dl_scores",
        )
        if vm_method == "PCA":
            vm_load = vm_pca.components_.T
            vm_load_df = pd.DataFrame(
                vm_load,
                index=vm_spec_cols_pca,
                columns=[f"PC{i}" for i in range(1, vm_n_comp + 1)],
            )
            if isinstance(vm_wls_pca, pd.Series) and not vm_wls_pca.empty:
                vm_load_df.insert(0, "wavelength", vm_wls_pca.values)
            st.download_button(
                "Loadings (CSV)",
                vm_load_df.to_csv(index=True).encode("utf-8"),
                "pca_loadings.csv",
                "text/csv",
                key="vm_dl_loadings",
            )

    # ------------------------------------------------------------
    # 4) VARIABILITÉ INTRA / INTER — ISOLÉE
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("Variabilité intra / inter (distance euclidienne)")

    try:
        from sklearn.metrics import pairwise_distances
    except Exception as e:
        st.error(f"scikit-learn requis (import échoué : {e})")
        st.stop()

    vm_spec_cols_dist, _ = _get_spectrum_cols(vm_out)
    if len(vm_spec_cols_dist) < 2:
        st.info("Distances indisponibles : moins de 2 colonnes spectrales détectées.")
        st.stop()

    # Attributs candidats
    vm_group_candidates = [
        c for c in ["Savonnette", "Lot", "Localisation", "Logo"] if c in vm_out.columns
    ]
    if not vm_group_candidates:
        st.info(
            "Aucun attribut de groupe trouvé parmi : Savonnette, Lot, Localisation, Logo."
        )
        st.stop()

    # ---- Contrôles
    c1, c2, c3 = st.columns(3)
    with c1:
        vm_intra_group_attr = st.selectbox(
            "Grouper (INTRA)",
            vm_group_candidates,
            index=(
                vm_group_candidates.index("Savonnette")
                if "Savonnette" in vm_group_candidates
                else 0
            ),
            key="vm_intra_group",
            help="Attribut utilisé pour les paires au sein d’un même groupe.",
        )
    with c2:
        vm_inter_group_attr = st.selectbox(
            "Grouper (INTER)",
            vm_group_candidates,
            index=(
                vm_group_candidates.index("Lot") if "Lot" in vm_group_candidates else 0
            ),
            key="vm_inter_group",
            help="Attribut utilisé pour comparer les groupes entre eux.",
        )
    with c3:
        vm_point_cap = st.number_input(
            "Sous-échantillon max. (paires affichées)",
            min_value=1000,
            max_value=1_000_000,
            value=200_000,
            step=1000,
            key="vm_var_cap",
            help="Sous-échantillonne aléatoirement les paires INTRA si nécessaire (affichage uniquement).",
        )

    colA, colB = st.columns(2)
    vm_sample_col2 = _find_col(["sample_name", "Sample"], vm_out.columns)
    with colA:
        vm_avg_before_dist = st.checkbox(
            "Moyenner les réplicas par sample (INTRA & INTER)",
            value=bool(vm_sample_col2),
            key="vm_var_avg",
        )
    with colB:
        vm_rep_mode = st.selectbox(
            "Représentant pour l’INTER",
            ["Centroïde (moyenne)", "Médoïde (échantillon le + proche du centre)"],
            index=1,
            key="vm_var_rep",
            help="Utilisé uniquement pour l’INTER (construction d’un représentant par groupe).",
        )

    # ---- Préparation commune
    vmD_base = vm_out.copy()
    if (
        "sample_name" not in vmD_base.columns
        and vm_sample_col2
        and vm_sample_col2 in vmD_base.columns
    ):
        vmD_base["sample_name"] = vmD_base[vm_sample_col2].astype(str)

    Xm_full = vmD_base[vm_spec_cols_dist].apply(pd.to_numeric, errors="coerce")
    mask_full = Xm_full.notna().all(axis=1)
    vmD_base = vmD_base.loc[mask_full].reset_index(drop=True)

    # Jeux séparés pour INTRA et INTER
    vmD_intra = vmD_base.dropna(subset=[vm_intra_group_attr]).reset_index(drop=True)
    vmD_inter = vmD_base.dropna(subset=[vm_inter_group_attr]).reset_index(drop=True)

    def _avg_by_sample(df: pd.DataFrame, group_attr: str) -> pd.DataFrame:
        """Moyenne des spectres par sample_name + attribut de groupe."""
        if "sample_name" not in df.columns:
            return df
        gb = ["sample_name", group_attr]
        return (
            df[gb + vm_spec_cols_dist]
            .groupby(gb, as_index=False)
            .mean(numeric_only=True)
        )

    if vm_avg_before_dist:
        vmD_intra = _avg_by_sample(vmD_intra, vm_intra_group_attr)
        vmD_inter = _avg_by_sample(vmD_inter, vm_inter_group_attr)

    # ---------- INTRA (paires au sein de chaque groupe)
    vm_intra_parts, vm_group_sizes = [], {}
    rng = np.random.default_rng(42)

    for g, sub in vmD_intra.groupby(vm_intra_group_attr):
        M = sub[vm_spec_cols_dist].to_numpy()
        n = M.shape[0]
        vm_group_sizes[g] = n
        if n < 2:
            continue
        D = pairwise_distances(M, metric="euclidean")
        tri = np.triu_indices(n, 1)
        vals = D[tri]
        if vals.size:
            vm_intra_parts.append(
                pd.DataFrame({"distance": vals, "type": "intra", "groupe": str(g)})
            )

    vm_intra_df = (
        pd.concat(vm_intra_parts, ignore_index=True)
        if vm_intra_parts
        else pd.DataFrame(columns=["distance", "type", "groupe"])
    )

    # Sous-échantillonnage d'affichage pour l'INTRA
    vm_intra_view = vm_intra_df
    if not vm_intra_df.empty and len(vm_intra_df) > vm_point_cap:
        idx = rng.choice(len(vm_intra_df), size=vm_point_cap, replace=False)
        vm_intra_view = vm_intra_df.iloc[idx].reset_index(drop=True)

    # ---------- INTER (distance entre représentants de groupes)
    reps = []
    for g, sub in vmD_inter.groupby(vm_inter_group_attr):
        M = sub[vm_spec_cols_dist].to_numpy()
        if M.size == 0:
            continue
        if vm_rep_mode.startswith("Centroïde"):
            rep = M.mean(axis=0, keepdims=True)
        else:
            centroid = M.mean(axis=0, keepdims=True)
            d = pairwise_distances(M, centroid, metric="euclidean").ravel()
            rep = M[np.argmin(d)][None, :]
        reps.append((str(g), rep))

    if len(reps) >= 2:
        labels = [g for g, _ in reps]
        R = np.vstack([r for _, r in reps])
        Dg = pairwise_distances(R, metric="euclidean")
        tri = np.triu_indices(len(reps), 1)
        vm_inter_df = pd.DataFrame(
            {
                "distance": Dg[tri],
                "type": "inter",
                "groupe_pair": [f"{labels[i]} ⟷ {labels[j]}" for i, j in zip(*tri)],
            }
        )
    else:
        vm_inter_df = pd.DataFrame(columns=["distance", "type", "groupe_pair"])

    # ---------- Fusion pour viz + stats
    vm_both = pd.concat(
        [
            vm_intra_view[["distance", "type"]].assign(
                detail=vm_intra_view.get("groupe")
            ),
            vm_inter_df[["distance", "type"]].assign(
                detail=vm_inter_df.get("groupe_pair")
            ),
        ],
        ignore_index=True,
    )

    n_by_type = vm_both.groupby("type")["distance"].count().to_dict()
    lbl_map = {
        "intra": f"Intra variability (n={n_by_type.get('intra', 0):,})",
        "inter": f"Inter variability (n={n_by_type.get('inter', 0):,})",
    }
    vm_both["type_label"] = vm_both["type"].map(lbl_map)

    # Récapitulatif chiffré
    def _q1(s: pd.Series) -> float:
        return float(np.nanquantile(s, 0.25)) if s.count() else np.nan

    def _q3(s: pd.Series) -> float:
        return float(np.nanquantile(s, 0.75)) if s.count() else np.nan

    vm_recap = (
        vm_both.groupby("type")["distance"]
        .agg(
            n="count",
            min="min",
            q1=_q1,
            median="median",
            q3=_q3,
            max="max",
            mean="mean",
            std="std",
        )
        .reset_index()
    )

    # ---------- Histogrammes côte à côte
    if not vm_both.empty:
        dmin, dmax = float(vm_both["distance"].min()), float(vm_both["distance"].max())
        if dmax <= dmin:
            dmax = dmin + 1e-9

        edges = np.linspace(dmin, dmax, 51)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]

        rows = []
        for lbl, sub in vm_both.groupby("type_label"):
            vals = sub["distance"].to_numpy()
            n = max(1, len(vals))
            cnt, _ = np.histogram(vals, bins=edges)
            pct = cnt / n * 100.0
            rows.append(
                pd.DataFrame(
                    {
                        "bin_label": bin_labels,
                        "bin_center": centers,
                        "pct": pct,
                        "type_label": lbl,
                    }
                )
            )
        vm_hist = pd.concat(rows, ignore_index=True)

        means = (
            vm_both.groupby("type_label", as_index=False)["distance"]
            .mean()
            .rename(columns={"distance": "mean"})
        )

        def nearest_label(v: float) -> str:
            idx = int(np.clip(np.searchsorted(centers, v), 1, len(centers)) - 1)
            return bin_labels[idx]

        means["mean_label"] = means["mean"].apply(nearest_label)
        ymax = (
            vm_hist.groupby("type_label", as_index=False)["pct"]
            .max()
            .rename(columns={"pct": "ymax"})
        )
        means = means.merge(ymax, on="type_label", how="left")
        means["label_y"] = means["ymax"].fillna(0) + 2

        bars = (
            alt.Chart(vm_hist)
            .mark_bar(size=12)
            .encode(
                x=alt.X(
                    "bin_label:N",
                    sort=bin_labels,
                    title="Euclidean Distance (binned)",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("pct:Q", title="Frequency (%)"),
                color=alt.Color("type_label:N", title="Legend"),
                xOffset=alt.XOffset("type_label:N"),
                tooltip=[
                    alt.Tooltip("type_label:N", title="Type"),
                    alt.Tooltip(
                        "bin_center:Q", title="Distance (center)", format=".2f"
                    ),
                    alt.Tooltip("pct:Q", title="Frequency (%)", format=".2f"),
                ],
            )
            .properties(height=340)
        )
        rules = (
            alt.Chart(means)
            .mark_rule(strokeDash=[6, 4])
            .encode(
                x=alt.X("mean_label:N", sort=bin_labels),
                color=alt.Color("type_label:N", legend=None),
            )
        )
        texts = (
            alt.Chart(means)
            .mark_text(dy=-6)
            .encode(
                x=alt.X("mean_label:N", sort=bin_labels),
                y=alt.Y("label_y:Q"),
                text=alt.Text("type_label:N"),
                color=alt.Color("type_label:N", legend=None),
            )
        )

        st.altair_chart(bars + rules + texts, use_container_width=True)
    else:
        st.info("Aucune distance à afficher.")

    # ---------- Boxplot comparatif
    if not vm_both.empty:
        box = (
            alt.Chart(vm_both)
            .mark_boxplot(size=60)
            .encode(
                x=alt.X("type_label:N", title=None),
                y=alt.Y("distance:Q", title="Euclidean Distance"),
                color=alt.Color("type_label:N", legend=None),
                tooltip=[alt.Tooltip("type_label:N"), alt.Tooltip("distance:Q")],
            )
            .properties(height=220)
        )
        st.altair_chart(box, use_container_width=True)

    # ---------- Tailles des groupes (INTRA)
    if "vm_group_sizes" in locals() and vm_group_sizes:
        vm_sizes_df = pd.DataFrame(
            sorted(vm_group_sizes.items(), key=lambda x: (-x[1], str(x[0]))),
            columns=[vm_intra_group_attr, "taille"],
        )
        with st.expander("Tailles des groupes (INTRA)"):
            st.dataframe(vm_sizes_df, use_container_width=True, height=260)

    # ---------- Exports variabilité
    with st.expander("⬇️ Exporter"):
        parts = []
        if "vm_intra_df" in locals() and not vm_intra_df.empty:
            intra_exp = (
                vm_intra_df.rename(columns={"groupe": "detail"})
                if "groupe" in vm_intra_df.columns
                else vm_intra_df.assign(detail=np.nan)
            )
            parts.append(intra_exp.loc[:, ["distance", "type", "detail"]])
        if "vm_inter_df" in locals() and not vm_inter_df.empty:
            inter_exp = (
                vm_inter_df.rename(columns={"groupe_pair": "detail"})
                if "groupe_pair" in vm_inter_df.columns
                else vm_inter_df.assign(detail=np.nan)
            )
            parts.append(inter_exp.loc[:, ["distance", "type", "detail"]])

        export_df = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["distance", "type", "detail"])
        )

        st.download_button(
            "Distances (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            f"distances_intra_{vm_intra_group_attr.lower()}__inter_{vm_inter_group_attr.lower()}.csv",
            "text/csv",
            key="vm_var_dl_dist",
        )
        st.download_button(
            "Résumé (CSV)",
            vm_recap.to_csv(index=False).encode("utf-8"),
            f"resume_intra_{vm_intra_group_attr.lower()}__inter_{vm_inter_group_attr.lower()}.csv",
            "text/csv",
            key="vm_var_dl_summary",
        )
