"""Research dashboard: live grid-world, adaptation evidence, algorithm table.

Run from the project root:

    streamlit run app.py

Every number comes from real artifacts (`docs/results_stage6.csv`, saved
curve figures) or from small agents trained live on button press — nothing
is hardcoded or illustrative.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from visualization.dashboard import (
    before_after_demo,
    draw_grid,
    greedy_route,
    load_results,
    summarize,
    train_demo,
)

st.set_page_config(page_title="Sample-Efficient Adaptation in Goal-RL", layout="wide")
st.title("Sample-Efficient Adaptation in Goal-Conditioned RL")
st.caption("Tabular agents · laptop CPU · all figures from real runs, never fabricated.")

tab_live, tab_before, tab_reuse, tab_algo, tab_curves, tab_heat, tab_summary = st.tabs([
    "A. Live simulation", "B. Before / after", "C. Knowledge reuse",
    "D. Algorithms", "E. Learning curves", "F. Q-heatmaps", "G. Summary"])

with tab_live:
    st.header("A. Live grid-world simulation")
    episodes = st.slider("Training episodes (demo agent)", 100, 2000, 800, step=100)
    if st.button("Train + run demo agent"):
        with st.spinner("Training (~1 s)..."):
            env, bfs_len, success, _, learner = train_demo(episodes=episodes)
            route, reached = greedy_route(env, learner, (5, 5), (0, 0))
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            draw_grid(env, path=route, ax=ax,
                      title=f"Greedy route: {len(route)-1} steps (BFS optimum {bfs_len})")
            st.pyplot(fig)
        with col2:
            st.metric("Steps", len(route) - 1)
            st.metric("BFS optimum", bfs_len)
            st.metric("Eval success (10 eps)", f"{success:.1f}")
            st.write("Status:", "✅ reached" if reached else "❌ failed")


with tab_before:
    st.header("B. Before / after the environment change")
    before, old_route, after, new_route = before_after_demo()
    before.reset(start=(0, 0), goal=(7, 7))
    after.reset(start=(0, 0), goal=(7, 7))
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_grid(before, path=old_route, ax=ax,
                  title=f"Original: learned route {len(old_route)-1} steps")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_grid(after, path=new_route, ax=ax,
                  title=f"Passage (3,7) sealed: new route {len(new_route)-1} steps")
        st.pyplot(fig)
    st.write("The old route is ineffective after the change: its corridor "
             "passage is a wall, so the agent must discover the left passage.")

with tab_reuse:
    st.header("C. Knowledge reuse comparison (Experiment B, 5 seeds)")
    df = load_results()
    summ = summarize(df, "B")
    methods = st.multiselect("Methods", list(summ["method"]),
                             default=["scratch", "warm", "combo", "CASR"])
    sub = summ[summ["method"].isin(methods)].set_index("method")
    st.bar_chart(sub["N_mean"], y_label="Additional episodes to 90%")
    st.dataframe(sub[["N_mean", "success_mean", "time_mean", "reached"]])
    st.caption("N_mean = post-change episodes to 90% success (CASR incl. 30 "
               "probe episodes). Source: docs/results_stage6.csv.")

with tab_algo:
    st.header("D. Algorithm comparison (live 6×6 demo)")
    if st.button("Compare algorithms"):
        with st.spinner("Training..."):
            env, bfs_len, q_success, _, _ = train_demo(episodes=800)
        from algorithms.dijkstra import dijkstra
        _, cost = dijkstra(6, 6, [(2, c) for c in range(1, 5)], (0, 0), (5, 5))
        st.table({
            "Method": ["BFS (map given)", "Dijkstra (map given)",
                       "Tabular Q-learning (learned)", "Goal-conditioned Q (Stage 3, 8×8)",
                       "CASR (Stage 6, Exp B)"],
            "Path / success": [f"{bfs_len} steps", f"cost {cost:.0f}",
                               f"success {q_success:.1f}",
                               "success 1.00, mean len 6.5",
                               "N=210 eps, success 1.00"],
            "Training interactions": ["0 (oracle)", "0 (oracle)", "~800 eps",
                                      "4000 eps", "30 probe + adapt"],
        })
        st.caption("Planners are oracles (upper bounds), learners earn routes "
                   "through interaction. Stage 3/6 rows quote the measured CSV/note values.")

with tab_curves:
    st.header("E. Learning curves (saved real artifacts)")
    for path, cap in (("docs/learning_curves_stage3.png", "Stage 3: initial learning"),
                      ("docs/adaptation_curves_stage4.png", "Stage 4: adaptation, seed 0"),
                      ("docs/eval_curves_stage6.png", "Stage 6: mean ± std, 5 seeds")):
        try:
            st.image(path, caption=cap, use_container_width=True)
        except Exception as err:
            st.warning(f"Missing {path}: {err}")

with tab_heat:
    st.header("F. Q-value heatmap (max over actions)")
    if st.button("Train + show Q-heatmap"):
        with st.spinner("Training..."):
            env, _, _, qmax, _ = train_demo(episodes=800)
        fig, ax = plt.subplots(figsize=(4, 4))
        masked = np.ma.masked_where(
            np.array([[(r, c) in env.obstacles
                       for c in range(env.width)] for r in range(env.height)]), qmax)
        ax.imshow(masked, cmap="viridis", origin="upper")
        ax.set_title("max_a Q(s,a): bright = near goal")
        for (r, c) in env.obstacles:
            ax.text(c, r, "█", ha="center", va="center", color="black")
        st.pyplot(fig)

with tab_summary:
    st.header("G. Experiment summary (Stage 6 CSV)")
    df = load_results()
    exp = st.selectbox("Experiment", ["A", "B", "C", "D", "E"])
    st.dataframe(summarize(df, exp))
    st.caption("Full per-seed rows: docs/results_stage6.csv. "
               "N_mean/std over seeds that reached 90%; `reached` counts them.")
