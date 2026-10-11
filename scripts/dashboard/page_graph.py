import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common as C

df = C.load_data()
summary = C.load_result("g01_graph_summary")

C.page_header(
    "Analytics · graph", "Graph Analytics",
    "The 1 km grid as a **graph**: each of the 9,333 cells is a node and cells that share a side are linked (rook "
    "neighbours). Graph algorithms find what a cell-by-cell table cannot: contiguous farmable land and the cells that hold "
    "it together, regions of similar growing conditions (Louvain communities), and which crops compete for the same land. "
    "Notebook `scripts/modeling/g01_graph_analytics.ipynb`, which also runs the graph on Apache Spark with GraphFrames.")
if summary is None:
    C.no_results_note("g01_graph_analytics.ipynb")
    st.stop()

S = summary.set_index("measure")["value"]


def value(key) -> float:
    try:
        return float(S[key])
    except (KeyError, ValueError):
        return np.nan


def num(key, fmt="{:,.0f}") -> str:
    v = value(key)
    return fmt.format(v) if np.isfinite(v) else str(S.get(key, "-"))


C.stat_strip([("Nodes (1 km cells)", num("nodes (1 km cells)")), ("Edges (shared sides)", num("edges (shared cell sides)")),
              ("Land masses", num("connected components (land masses)")), ("Louvain communities", num("Louvain communities")),
              ("Modularity", num("modularity", "{:.3f}")), ("GraphFrames = NetworkX", num("GraphFrames checks passed"))])


@st.cache_resource(show_spinner=False)
def grid_graph() -> nx.Graph:
    """Rook-adjacency graph of the 1 km cells; node i is row i of the fact table (and of load_data())."""
    d = pd.read_csv(C.FEATURES_CSV, usecols=["row", "col"])
    where = {(r, c): i for i, (r, c) in enumerate(zip(d["row"], d["col"]))}
    G = nx.Graph()
    G.add_nodes_from(range(len(d)))
    G.add_edges_from((i, where[(r + dr, c + dc)]) for (r, c), i in where.items()
                     for dr, dc in ((1, 0), (0, 1)) if (r + dr, c + dc) in where)
    return G


@st.cache_data(show_spinner=False, max_entries=128)
def patch_ids(mask: np.ndarray) -> np.ndarray:
    """Connected patch of every selected cell, numbered from the largest (0) down; -1 for cells not selected."""
    comps = sorted(nx.connected_components(grid_graph().subgraph(np.flatnonzero(mask).tolist())), key=len, reverse=True)
    ids = np.full(len(mask), -1)
    for k, comp in enumerate(comps):
        ids[list(comp)] = k
    return ids


@st.cache_data(show_spinner=False, max_entries=32)
def bottlenecks(mask: np.ndarray) -> list[int]:
    """Articulation points: cells whose loss would split the patch in two."""
    return sorted(nx.articulation_points(grid_graph().subgraph(np.flatnonzero(mask).tolist())))


def patch_sizes(ids: np.ndarray) -> np.ndarray:
    return np.bincount(ids[ids >= 0]) if (ids >= 0).any() else np.zeros(0, dtype=int)


def size_per_cell(ids: np.ndarray) -> np.ndarray:
    """Size of the patch each cell belongs to (NaN for cells outside every patch)."""
    sizes = patch_sizes(ids)
    return np.where(ids >= 0, sizes[np.maximum(ids, 0)] if len(sizes) else 0, np.nan)


def patch_stats(ids: np.ndarray) -> dict:
    sizes = patch_sizes(ids)
    total = int(sizes.sum())
    return {"suitable km²": total, "patches": len(sizes), "largest patch km²": int(sizes.max()) if total else 0,
            "in largest patch": sizes.max() / total if total else np.nan,
            "in patches ≥ 25 km²": sizes[sizes >= 25].sum() / total if total else np.nan,
            "single-cell patches": int((sizes == 1).sum())}


with st.container(border=True):
    a, b = st.columns([1, 2])
    threshold = a.slider("Suitable from (GAEZ index)", 0, 10000, C.NOT_RECOMMENDED_THRESHOLD, 250, key="graph_threshold",
                         help="A cell counts as suitable land for a crop when its GAEZ index reaches this value "
                              "(5,000 = the rule of the recommendation).")
    b.caption(f"Used by the tabs 'Contiguous suitable land' and 'Crop competition', with GAEZ at {C.gaez_res()} (switch "
              "in the sidebar); both recompute the graph live. The notebook works at 1 km and 5,000.")
masks = {c: (df[C.gaez_col(c)] >= threshold).to_numpy() for c in C.CROPS}
tab_patch, tab_block, tab_comm, tab_comp, tab_spark = st.tabs(
    ["Contiguous suitable land", "Recommendation blocks", "Agro-ecological communities", "Crop competition",
     "Spark GraphFrames"])

# ------------------------------------------------------------------------------------------------ patches
with tab_patch:
    C.section("Contiguous suitable land", "connected components of the suitable cells: a patch is land you can cross "
                                          "without leaving suitable ground")
    crop = st.selectbox("Crop", C.CROPS, format_func=C.CROP_LABELS.get, key="graph_crop")
    ids = patch_ids(masks[crop])
    stats = patch_stats(ids)
    cut = bottlenecks(ids == 0) if stats["patches"] else []
    C.stat_strip([("Suitable land", f"{stats['suitable km²']:,} km²"), ("Patches", f"{stats['patches']:,}"),
                  ("Largest patch", f"{stats['largest patch km²']:,} km²"),
                  ("Share in largest", f"{stats['in largest patch']:.0%}" if stats["patches"] else "-"),
                  ("Single-cell patches", f"{stats['single-cell patches']:,}"), ("Bottleneck cells", f"{len(cut):,}")])
    view = st.segmented_control("Show", ["Patch size", "Largest patch and bottlenecks"], default="Patch size",
                                key="graph_patch_view") or "Patch size"
    left, right = st.columns([1.3, 1])
    with left:
        if view == "Patch size":
            fig = C.grid_map(df, np.log10(size_per_cell(ids)), f"{C.CROP_LABELS[crop]}: suitable cells by the size of "
                             "their patch", C.SEQ, 0, np.log10(max(stats["largest patch km²"], 10)), 470, "log10 km²")
        else:
            role = pd.Series(np.select([ids == 0, ids > 0], ["Largest patch", "Other patches"], "Not suitable"), index=df.index)
            role.iloc[cut] = "Bottleneck cell"
            fig = C.category_map(df, role, {"Largest patch": C.PINE, "Bottleneck cell": C.CLAY, "Other patches": C.SAGE,
                                            "Not suitable": "#ECEAE3"}, f"Largest {C.CROP_LABELS[crop].lower()} patch", 470)
        st.plotly_chart(fig, width="stretch")
    with right:
        frag = pd.DataFrame({C.CROP_LABELS[c]: patch_stats(patch_ids(masks[c])) for c in C.CROPS}).T
        big = frag["in patches ≥ 25 km²"].fillna(0) * frag["suitable km²"]
        parts = pd.DataFrame({"largest patch": frag["largest patch km²"],
                              "other patches ≥ 25 km²": np.maximum(big - frag["largest patch km²"], 0),
                              "patches < 25 km²": frag["suitable km²"] - np.maximum(big, frag["largest patch km²"])})
        long = parts.rename_axis("crop").reset_index().melt(id_vars="crop", var_name="part", value_name="km²")
        fig = px.bar(long, x="km²", y="crop", color="part", orientation="h", title="How the suitable land is split up",
                     color_discrete_sequence=[C.PINE, C.SAGE, C.OCHRE],
                     category_orders={"crop": frag.sort_values("suitable km²", ascending=False).index.tolist()})
        fig.update_layout(height=470, margin=dict(t=40), yaxis_title="", legend=dict(orientation="h", y=-0.15, title=""))
        st.plotly_chart(fig, width="stretch")
    C.note("Bottleneck cells are the articulation points of the largest patch: remove one and the patch falls apart. "
           "They mark narrow corridors of suitable land, where a road, a settlement or a change of land use would cut the "
           "farmable area in two.")
    with st.expander("All crops as a table"):
        st.dataframe(frag.round(3), width="stretch")

# ------------------------------------------------------------------------------------------------ recommendation blocks
with tab_block:
    group, ranking = C.rule()
    C.section("Contiguous blocks of the recommended crop", f"{group} · {ranking.lower()} ranking · GAEZ {C.gaez_res()} "
                                                           "(sidebar) · colour depth = size of the block")
    label = df["recommended_crop"]
    block_size, rows = np.full(len(df), np.nan), []
    for lab in C.recommendation_options(df):
        if lab in C.FLAT_LABELS:
            continue
        ids_l = patch_ids((label == lab).to_numpy())
        sizes = patch_sizes(ids_l)
        block_size[ids_l >= 0] = sizes[ids_l[ids_l >= 0]]
        rows.append({"recommended crop": lab, "cells": int(sizes.sum()), "blocks": len(sizes), "largest block km²": int(sizes.max()),
                     "mean block km²": sizes.mean(), "in blocks ≥ 25 km²": sizes[sizes >= 25].sum() / sizes.sum()})
    if not rows:
        st.info("No cell has a recommended crop under the current rule.")
    else:
        blocks = pd.DataFrame(rows).sort_values("cells", ascending=False)
        depth = np.log10(np.nan_to_num(block_size, nan=1.0)) / np.log10(max(np.nanmax(block_size), 10))
        hover = label + "<br>" + pd.Series(block_size, index=df.index).map(lambda v: "" if np.isnan(v) else f"block of {v:,.0f} km²")
        left, right = st.columns([1.3, 1])
        left.plotly_chart(C.category_map(df, label, C.LABEL_COLORS, height=470, strength=0.15 + 0.85 * depth, hover=hover),
                          width="stretch")
        with right:
            st.dataframe(blocks.round(2), width="stretch", hide_index=True)
            st.caption("A block is a set of neighbouring cells with the same recommendation. Large blocks suit one estate, "
                       "a cooperative or one irrigation scheme; many small blocks mean a mosaic where no single crop "
                       "dominates a whole area.")

# ------------------------------------------------------------------------------------------------ communities
with tab_comm:
    C.section("Agro-ecological communities", "Louvain on the grid graph, edges weighted by how similar the climate, soil "
                                              "and terrain of the two cells are (notebook g01, 1 km)")
    nodes_path = C.OOF_DIR / "g01_nodes.parquet"
    prof = C.load_result("g01_communities")
    if not nodes_path.exists() or prof is None:
        C.no_results_note("g01_graph_analytics.ipynb")
    else:
        prof = prof.set_index("community")
        nodes = df[["grid_id"]].merge(pd.read_parquet(nodes_path, columns=["grid_id", "community"]), on="grid_id", how="left")
        com = nodes["community"].fillna(0).astype(int).to_numpy()
        G = grid_graph()
        A = nx.Graph()
        A.add_nodes_from(prof.index)
        A.add_edges_from((com[u], com[v]) for u, v in G.edges if com[u] != com[v] and com[u] and com[v])
        colour = nx.greedy_color(A, strategy="largest_first")   # neighbouring communities never share a colour
        pal = [C.PINE, C.OCHRE, C.TEAL, C.CLAY, C.MOSS, C.BARK, "#8DB474", "#73B0A6", "#D9AE87", "#9C8B55"]
        picked = st.selectbox("Highlight a community", ["All"] + prof.index.tolist(), key="graph_comm",
                              format_func=lambda k: k if k == "All" else f"{k} · {prof.loc[k, 'cells']:,} km² · mostly "
                                                                         f"{prof.loc[k, 'most frequent recommendation']}")
        names = pd.Series(com).map(lambda k: f"Community {k}")
        colors = {f"Community {k}": pal[colour.get(k, 0) % len(pal)] for k in prof.index}
        if picked != "All":
            names = names.where(com == picked, "Other communities")
            colors = {f"Community {picked}": C.PINE, "Other communities": "#ECEAE3"}
        info = {k: f"Community {k} · {p['cells']:,} km²<br>{p['elevation_m']:,.0f} m · {p['rainfall_mm']:,.0f} mm · "
                   f"{p['temperature_c']:.1f} °C<br>most often {p['most frequent recommendation']} ({p['its share']:.0%})"
                for k, p in prof.iterrows()}
        fig = C.category_map(df, names.set_axis(df.index), colors, height=470, hover=pd.Series(com).map(info).fillna("").to_numpy())
        fig.update_traces(showscale=False)
        left, right = st.columns([1.3, 1])
        left.plotly_chart(fig, width="stretch")
        right.dataframe(prof.round(1), width="stretch", height=470)
        nmi_label, nmi_lc = value("NMI community vs recommended crop"), value("NMI community vs land cover")
        C.note(f"Modularity {num('modularity', '{:.2f}')} (0 = no better than a random split; above about 0.3 = clear "
               f"community structure). Agreement of the communities with the recommendation map, as normalised mutual "
               f"information, is {nmi_label:.2f}, with today's land cover {nmi_lc:.2f}: the communities follow "
               + ("the growing conditions that shape the crop map more than the current land use. " if nmi_label > nmi_lc
                  else "current land use at least as much as the crop map. ")
               + "m07 (ML/DL Experiments) builds zones from the same drivers with clustering instead of a graph.")

# ------------------------------------------------------------------------------------------------ crop competition
with tab_comp:
    C.section("Crop competition", "on which land do the crops compete? suitable = GAEZ index at or above the threshold "
                                  "above; node size = suitable land")
    area = {c: int(masks[c].sum()) for c in C.CROPS}
    inter = {(a_, b_): int((masks[a_] & masks[b_]).sum()) for a_ in C.CROPS for b_ in C.CROPS}
    union = {(a_, b_): int((masks[a_] | masks[b_]).sum()) for a_ in C.CROPS for b_ in C.CROPS}
    top = max(area.values()) or 1
    names = [C.CROP_LABELS[c] for c in C.CROPS]

    def bubble(n_area):
        return 14 + 40 * np.sqrt(n_area / top)

    # Overlap: undirected, weighted by the Jaccard index.
    CG = nx.Graph()
    CG.add_nodes_from(C.CROPS)
    CG.add_weighted_edges_from((a_, b_, inter[a_, b_] / union[a_, b_]) for i, a_ in enumerate(C.CROPS)
                               for b_ in C.CROPS[i + 1:] if union[a_, b_] and inter[a_, b_] / union[a_, b_] >= 0.05)
    pos = nx.circular_layout(CG)
    strength = dict(CG.degree(weight="weight"))
    net = go.Figure()
    for a_, b_, w in CG.edges(data="weight"):
        net.add_trace(go.Scatter(x=[pos[a_][0], pos[b_][0]], y=[pos[a_][1], pos[b_][1]], mode="lines", hoverinfo="skip",
                                 line=dict(width=1 + 14 * w, color="rgba(110,128,106,%.2f)" % (0.25 + 0.6 * w))))
    net.add_trace(go.Scatter(x=[pos[c][0] for c in C.CROPS], y=[pos[c][1] for c in C.CROPS], mode="markers+text",
                             text=names, textposition="top center",
                             marker=dict(size=[bubble(area[c]) for c in C.CROPS], line=dict(color="white", width=1.5),
                                         color=[C.LABEL_COLORS[C.CROP_LABELS[c]] for c in C.CROPS]),
                             customdata=[[area[c], strength[c]] for c in C.CROPS],
                             hovertemplate="%{text}<br>suitable %{customdata[0]:,} km²<br>overlap strength "
                                           "%{customdata[1]:.2f}<extra></extra>"))
    net.update_layout(title="Overlap: edge width = Jaccard index (from 0.05)")

    # Nesting: A -> B when at least 95 % of A's suitable land is suitable for B; drawn as its Hasse diagram.
    D = nx.DiGraph()
    D.add_nodes_from(c for c in C.CROPS if area[c])
    D.add_edges_from((a_, b_) for a_ in D for b_ in D if a_ != b_ and inter[a_, b_] >= 0.95 * area[a_])
    Dc = nx.condensation(D)
    H = nx.transitive_reduction(Dc)
    member = {n: sorted(Dc.nodes[n]["members"], key=C.CROPS.index) for n in Dc}
    label = {n: " = ".join(C.CROP_LABELS[c] for c in member[n]) for n in Dc}
    n_area = {n: int(np.logical_or.reduce([masks[c] for c in member[n]]).sum()) for n in Dc}
    gens = list(nx.topological_generations(H))
    hpos = {n: (k - (len(g) - 1) / 2, i) for i, g in enumerate(gens)
            for k, n in enumerate(sorted(g, key=lambda n: C.CROPS.index(member[n][0])))}
    hasse = go.Figure()
    for u, v in H.edges:
        hasse.add_annotation(x=hpos[v][0], y=hpos[v][1], ax=hpos[u][0], ay=hpos[u][1], xref="x", yref="y", axref="x",
                             ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.6, arrowcolor="#8A9A86",
                             standoff=bubble(n_area[v]) / 2 + 2, startstandoff=bubble(n_area[u]) / 2 + 2, text="")
    hasse.add_trace(go.Scatter(x=[hpos[n][0] for n in H], y=[hpos[n][1] for n in H], mode="markers+text",
                               text=[label[n] for n in H], textposition="bottom center",
                               marker=dict(size=[bubble(n_area[n]) for n in H], line=dict(color="white", width=1.5),
                                           color=[C.LABEL_COLORS[C.CROP_LABELS[member[n][0]]] for n in H]),
                               customdata=[n_area[n] for n in H],
                               hovertemplate="%{text}<br>suitable %{customdata:,} km²<extra></extra>"))
    hasse.update_layout(title="Nesting: arrow A → B = 95 % of A's suitable land is suitable for B")
    for fig in (net, hasse):
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=500, showlegend=False, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    net.update_yaxes(scaleanchor="x")
    left, right = st.columns(2)
    left.plotly_chart(net, width="stretch")
    if H.number_of_nodes():
        right.plotly_chart(hasse, width="stretch")
    else:
        right.info("No crop reaches this threshold anywhere.")

    which = st.segmented_control("Matrix", ["Jaccard overlap", "Share of the row crop's land suitable for the column crop"],
                                 default="Share of the row crop's land suitable for the column crop", key="graph_matrix")
    if which == "Jaccard overlap":
        mat = pd.DataFrame([[inter[a_, b_] / union[a_, b_] if union[a_, b_] else 0.0 for b_ in C.CROPS] for a_ in C.CROPS],
                           index=names, columns=names)
    else:
        mat = pd.DataFrame([[inter[a_, b_] / area[a_] if area[a_] else np.nan for b_ in C.CROPS] for a_ in C.CROPS],
                           index=names, columns=names)
    heat = px.imshow(mat.round(2), text_auto=True, zmin=0, zmax=1, color_continuous_scale=C.SEQ, aspect="auto")
    heat.update_layout(height=430, margin=dict(t=10), xaxis_title="", yaxis_title="")
    st.plotly_chart(heat, width="stretch")
    no_land = [C.CROP_LABELS[c] for c in C.CROPS if not area[c]]
    text = ""
    if D.number_of_nodes():
        widest = max(D.nodes, key=lambda c: (D.in_degree(c), area[c]))
        text += (f"{C.CROP_LABELS[widest]} is the most widely suitable crop: the suitable land of "
                 f"{D.in_degree(widest)} other crops lies (95 % or more) within its own. ")
        if D.number_of_edges() >= D.number_of_nodes():
            text += ("The suitable lands are largely nested rather than side by side, so the crops compete for one core of "
                     "good land and the recommendation rule decides which crop gets it. ")
        else:
            text += "Few crops are nested in one another: they largely divide the province between them. "
    if no_land:
        text += "No suitable land at this threshold: " + ", ".join(no_land) + "."
    if text:
        C.note(text)

# ------------------------------------------------------------------------------------------------ GraphFrames
with tab_spark:
    C.section("The same graph on Apache Spark", "GraphFrames: vertices and edges as Spark DataFrames, algorithms as "
                                                "distributed jobs (local mode, 6 cores)")
    check, times = C.load_result("g01_graphframes_check"), C.load_result("g01_graphframes_timings")
    if check is None:
        C.no_results_note("g01_graph_analytics.ipynb")
    else:
        left, right = st.columns([1.3, 1])
        left.dataframe(check, width="stretch", hide_index=True, height=470)
        if times is not None:
            long = times.melt(id_vars="step", var_name="library", value_name="seconds")
            long["library"] = long["library"].str.replace(" (s)", "", regex=False)
            fig = px.bar(long, x="seconds", y="step", color="library", orientation="h", barmode="group", log_x=True,
                         title="Run time, NetworkX vs GraphFrames (log scale)",
                         color_discrete_map={"NetworkX": C.SAGE, "GraphFrames": C.TEAL})
            fig.update_layout(height=470, margin=dict(t=40), yaxis_title="", legend=dict(orientation="h", y=-0.15, title=""))
            right.plotly_chart(fig, width="stretch")
        verdicts = check["equal"].dropna().astype(str).str.lower().eq("true")
        text = (f"All {len(verdicts)} checks agree: GraphFrames returns the same node degrees, land masses and suitable-land "
                "patches as NetworkX. " if verdicts.all() else
                f"{int(verdicts.sum())} of {len(verdicts)} checks agree with NetworkX; see the table. ")
        if times is not None:
            t_gf, t_nx = times["GraphFrames (s)"].sum(), times["NetworkX (s)"].sum()
            text += (f"On 9,333 nodes Spark is the slower engine ({t_gf:,.0f} s in total against {t_nx:,.1f} s for "
                     "NetworkX): starting jobs costs more than the work itself. " if t_gf > t_nx else
                     f"GraphFrames needed {t_gf:,.0f} s in total against {t_nx:,.1f} s for NetworkX. ")
        text += "Its value is that the same code runs unchanged on a cluster when the grid covers a whole country."
        C.note(text)
        lpa = C.load_result("g01_label_propagation")
        if lpa is not None:
            C.section("Communities on Spark: label propagation", "GraphFrames' scalable community method, run for a fixed "
                                                                  "number of rounds")
            l2, r2 = st.columns([1, 1.3])
            l2.dataframe(lpa.round(3), width="stretch", hide_index=True)
            first, last = lpa.iloc[0], lpa.iloc[-1]
            r2.caption(f"Louvain (NetworkX) finds {num('Louvain communities')} communities. Label propagation ignores the "
                       f"similarity weights and ends with {int(last['communities']):,} communities after "
                       f"{int(last['rounds'])} rounds ({int(first['communities']):,} after {int(first['rounds'])}); its "
                       f"agreement with Louvain, as normalised mutual information, is {last['NMI vs Louvain']:.2f}. It is a "
                       "fast first partition of a very large graph rather than a replacement for Louvain.")
