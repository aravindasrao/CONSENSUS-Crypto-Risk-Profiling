# src/visualization/forensic_visualizer.py
"""
Consolidated forensic visualization module.
Merges advanced plotting from pattern_cluster_visualizer with network, risk,
and temporal plots for a comprehensive visualization suite.
"""
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

class ForensicVisualizer:
    """A centralized class for creating all forensic visualizations."""

    def __init__(self, database: DatabaseEngine, results_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            database: An active DatabaseEngine instance.
        """
        self.database = database
        self.output_dir = results_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = px.colors.qualitative.Plotly
        logger.info(f"ForensicVisualizer initialized. Outputs will be saved to {self.output_dir}")

    def generate_all_visuals(self, top_n: int = 3):
        """
        Orchestration method to generate a full suite of key visualizations
        for the most relevant entities.
        
        Args:
            top_n: The number of top entities (addresses, clusters, paths) to visualize.
        """
        logger.info(f"--- Generating comprehensive visualization suite for top {top_n} entities ---")

        # 1. Visualize Riskiest Addresses
        top_addresses = self.database.fetch_df("SELECT address FROM addresses ORDER BY risk_score DESC NULLS LAST LIMIT ?", (top_n,))
        for addr in top_addresses['address']:
            self.plot_risk_contribution_bar_chart(address=addr)
        
        # 2. Visualize Riskiest/Largest Clusters
        top_clusters = self.database.fetch_df("""
            SELECT cluster_id FROM addresses WHERE cluster_id IS NOT NULL 
            GROUP BY cluster_id ORDER BY AVG(risk_score) DESC NULLS LAST LIMIT ?
        """, (top_n,))
        for cid in top_clusters['cluster_id']:
            self.save_cluster_network_graph_static(cluster_id=int(cid))

        # 3. Visualize a Top Suspicious Multi-Hop Path
        top_path = self.database.fetch_one("SELECT path_id FROM suspicious_paths ORDER BY suspicion_score DESC LIMIT 1")
        if top_path:
            self.plot_multi_hop_sankey(path_id=top_path['path_id'])

        # 4. Visualize Cluster Activity Timeline
        self.plot_cluster_activity_timeline(num_clusters=15)
        
        # 6. Visualize Temporal Anomalies
        self.plot_temporal_anomaly_chart()

        # +++ NEW VISUALIZATIONS +++
        # 7. Visualize Attribution Links
        self.plot_attribution_graph()

        # 8. Visualize Frequent Behavioral Patterns (TTPs)
        self.plot_frequent_patterns_barchart()

        # 9. Visualize Deposit/Withdrawal Pattern Clusters
        self.plot_pattern_cluster_dashboard(top_k=top_n)
        self.plot_pattern_similarity_heatmap(top_k=top_n)

        logger.info("--- Visualization suite generation complete ---")

    def plot_cluster_network_graph_interactive(self, cluster_id: int) -> go.Figure:
        """
        Generates an INTERACTIVE network graph for a cluster using Plotly.
        This is designed to be displayed in the web app.
        """
        logger.info(f"Generating INTERACTIVE network graph for cluster {cluster_id}...")
        try:
            txs = self.database.fetch_df("""
                SELECT t.from_addr, t.to_addr, t.value_eth
                FROM transactions t JOIN addresses a ON (t.from_addr = a.address OR t.to_addr = a.address)
                WHERE a.cluster_id = ?
            """, (cluster_id,))

            if txs.empty:
                return go.Figure().update_layout(title_text=f"No transactions for Cluster {cluster_id}", template="plotly_dark")

            # FIX: Drop rows with NaN/None addresses to prevent NetworkX error
            txs.dropna(subset=['from_addr', 'to_addr'], inplace=True)
            if txs.empty:
                logger.warning(f"No valid transaction edges for Cluster {cluster_id} after cleaning.")
                return go.Figure().update_layout(title_text=f"No valid transaction edges for Cluster {cluster_id}", template="plotly_dark")

            G = nx.from_pandas_edgelist(txs, 'from_addr', 'to_addr', ['value_eth'], create_using=nx.DiGraph())
            pos = nx.spring_layout(G, k=0.2, iterations=50)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x, node_y, node_text, node_size = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                degree = G.degree(node)
                node_size.append(10 + degree * 5)
                node_text.append(f"Address: {node}<br>Degree: {degree}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=node_size,
                    colorbar=dict(
                        thickness=15, 
                        title=dict(text='Node Connections', side='right'), 
                        xanchor='left'
                    ),
                    line_width=2))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title_text=f'Interactive Network Graph for Cluster {cluster_id}',
                            title_font_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            template="plotly_dark"
                        ))
            return fig
        except Exception as e:
            logger.error(f"Failed to create interactive graph for cluster {cluster_id}: {e}")
            return go.Figure().update_layout(title_text=f"Error generating graph for Cluster {cluster_id}", template="plotly_dark")

    ### ================================================= ###
    ### ADVANCED VISUALIZATIONS FOR STATIC REPORTS        ###
    ### ================================================= ###

    def generate_advanced_visuals(self, top_n_flows: int = 25):
        """
        Generates and saves a suite of advanced visualizations for static reports.
        This is called from the pipeline orchestrator during the finalization phase.
        """
        logger.info("--- Generating advanced static visualizations for reports ---")
        
        # Create a dedicated directory for these visuals
        advanced_viz_dir = self.output_dir / "advanced_static_visuals"
        advanced_viz_dir.mkdir(exist_ok=True)
        logger.info(f"Advanced visuals will be saved to: {advanced_viz_dir}")

        # 1. Generate and save a Sankey diagram for fund flows
        sankey_fig = self.create_sankey_for_top_flows(top_n_flows=top_n_flows)
        if sankey_fig:
            sankey_path = advanced_viz_dir / "inter_cluster_fund_flow.png"
            try:
                sankey_fig.write_image(sankey_path, width=1200, height=800, scale=2)
                logger.info(f"Saved Sankey diagram to {sankey_path}")
            except Exception as e:
                logger.error(f"Failed to save Sankey diagram image. Ensure 'kaleido' is installed (`pip install kaleido`). Error: {e}")

        # 2. Generate and save a temporal heatmap of all transaction activity
        heatmap_fig = self.create_temporal_heatmap()
        if heatmap_fig:
            heatmap_path = advanced_viz_dir / "temporal_activity_heatmap.png"
            try:
                heatmap_fig.write_image(heatmap_path, width=1000, height=600, scale=2)
                logger.info(f"Saved temporal heatmap to {heatmap_path}")
            except Exception as e:
                logger.error(f"Failed to save temporal heatmap image. Error: {e}")

        # 3. Generate and save a risk score distribution histogram
        risk_hist_fig = self.create_risk_distribution_histogram()
        if risk_hist_fig:
            risk_hist_path = advanced_viz_dir / "risk_score_distribution.png"
            try:
                risk_hist_fig.write_image(risk_hist_path, width=800, height=500, scale=2)
                logger.info(f"Saved risk score histogram to {risk_hist_path}")
            except Exception as e:
                logger.error(f"Failed to save risk score histogram image. Error: {e}")
        
        # 4. Generate and save a Risk vs. Value scatter plot
        risk_value_fig = self.plot_risk_vs_value_scatter()
        if risk_value_fig:
            risk_value_path = advanced_viz_dir / "risk_vs_value_scatter.png"
            try:
                risk_value_fig.write_image(risk_value_path, width=1000, height=600, scale=2)
                logger.info(f"Saved risk vs. value scatter plot to {risk_value_path}")
            except Exception as e:
                logger.error(f"Failed to save risk vs. value scatter plot image. Error: {e}")

        logger.info("--- Advanced static visualization generation complete ---")

    def create_sankey_for_top_flows(self, top_n_flows: int = 25) -> go.Figure:
        """
        Creates a Sankey diagram showing ETH flow between the largest clusters.
        This is a class method version of the one in web_visualizers.
        """
        logger.info("Generating inter-cluster Sankey diagram for static report...")
        try:
            query = f"""
            WITH cluster_transactions AS (
                SELECT
                    t.value_eth,
                    fa.cluster_id AS from_cluster,
                    ta.cluster_id AS to_cluster
                FROM transactions t
                JOIN addresses fa ON t.from_addr = fa.address
                JOIN addresses ta ON t.to_addr = ta.address
                WHERE fa.cluster_id IS NOT NULL
                  AND ta.cluster_id IS NOT NULL
                  AND fa.cluster_id != ta.cluster_id
            )
            SELECT
                from_cluster,
                to_cluster,
                SUM(value_eth) as total_volume
            FROM cluster_transactions
            GROUP BY from_cluster, to_cluster
            ORDER BY total_volume DESC
            LIMIT {top_n_flows};
            """
            flows_df = self.database.fetch_df(query)

            if flows_df.empty:
                logger.warning("No inter-cluster transaction data found for Sankey diagram.")
                return None

            all_clusters = pd.unique(flows_df[['from_cluster', 'to_cluster']].values.ravel('K'))
            cluster_map = {cluster: i for i, cluster in enumerate(all_clusters)}

            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[f"Cluster {c}" for c in all_clusters],
                    color="blue"
                ),
                link=dict(
                    source=[cluster_map[row['from_cluster']] for _, row in flows_df.iterrows()],
                    target=[cluster_map[row['to_cluster']] for _, row in flows_df.iterrows()],
                    value=[row['total_volume'] for _, row in flows_df.iterrows()]
                ))])

            fig.update_layout(
                title_text="Top Inter-Cluster Fund Flows (ETH)",
                font_size=12
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create Sankey diagram for static report: {e}")
            return None

    def create_temporal_heatmap(self) -> go.Figure:
        """
        Creates a heatmap of transaction activity by day of the week and hour.
        This version is optimized to perform aggregation in the database.
        """
        logger.info("Generating temporal activity heatmap for static report...")
        try:
            # This query performs the aggregation directly in DuckDB, which is much more memory-efficient.
            query = """
            SELECT
                dayname(to_timestamp(timestamp)) as day_of_week,
                hour(to_timestamp(timestamp)) as hour,
                SUM(value_eth) as total_volume
            FROM transactions
            GROUP BY 1, 2
            """
            agg_df = self.database.fetch_df(query)

            if agg_df.empty:
                logger.warning("No transaction data found for temporal heatmap.")
                return None

            # Now pivot the much smaller, pre-aggregated DataFrame
            heatmap_data = agg_df.pivot_table(
                index='day_of_week', columns='hour', values='total_volume'
            ).fillna(0)

            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = heatmap_data.reindex(days_order)

            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Total ETH Volume"),
                title="Total Transaction Volume by Day and Hour",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create temporal heatmap for static report: {e}")
            return None

    def create_risk_distribution_histogram(self) -> go.Figure:
        """
        Creates a histogram showing the distribution of final risk scores.
        """
        logger.info("Generating risk score distribution histogram for static report...")
        try:
            query = "SELECT risk_score FROM addresses WHERE risk_score IS NOT NULL"
            risk_df = self.database.fetch_df(query)

            if risk_df.empty:
                logger.warning("No risk score data found for histogram.")
                return None

            fig = px.histogram(
                risk_df, x="risk_score", nbins=50, title="Distribution of Final Address Risk Scores",
                labels={'risk_score': 'Final Risk Score'}
            )
            fig.update_layout(bargap=0.1)
            return fig
        except Exception as e:
            logger.error(f"Failed to create risk distribution histogram for static report: {e}")
            return None

    ### ================================================= ###
    ### NEW VISUALIZATION: RISK VS. VALUE SCATTER PLOT    ###
    ### ================================================= ###

    def plot_risk_vs_value_scatter(self) -> go.Figure:
        """
        Creates a scatter plot of address risk score vs. average transaction value.
        This helps visualize the 'U-shaped' curve of risk.
        """
        logger.info("Generating risk vs. value scatter plot for static report...")
        try:
            query = """
            SELECT
                risk_score,
                total_volume_eth / total_transaction_count AS avg_tx_value,
                is_tornado
            FROM addresses
            WHERE risk_score IS NOT NULL AND total_transaction_count > 0 AND total_volume_eth > 0;
            """
            df = self.database.fetch_df(query)

            if df.empty:
                logger.warning("No data found for risk vs. value plot.")
                return None

            # Add a small amount to avg_tx_value to avoid log(0) issues
            df['avg_tx_value'] = df['avg_tx_value'] + 1e-9

            fig = px.scatter(
                df,
                x="avg_tx_value",
                y="risk_score",
                color="is_tornado",
                log_x=True,
                title="Address Risk Score vs. Average Transaction Value (Log Scale)",
                labels={'avg_tx_value': 'Average Transaction Value (ETH)', 'risk_score': 'Final Risk Score', 'is_tornado': 'Tornado User'},
                hover_data={'avg_tx_value': ':.4f'},
                template="plotly_dark",
                color_discrete_map={True: 'red', False: 'cornflowerblue'},
                opacity=0.7
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create risk vs. value scatter plot: {e}")
            return None

    ### ================================================= ###
    ### VISUALIZATIONS FOR GENERAL FORENSIC ANALYSIS      ###
    ### ================================================= ###

    def save_cluster_network_graph_static(self, cluster_id: int):
        """Saves a STATIC network graph of a cluster to a file using Matplotlib."""
        # This is your original matplotlib-based function, now renamed for clarity
        logger.info(f"Saving STATIC network graph for cluster {cluster_id}...")
        try:
            txs = self.database.fetch_df("""
                SELECT t.from_addr, t.to_addr, t.value_eth
                FROM transactions t JOIN addresses a ON (t.from_addr = a.address OR t.to_addr = a.address)
                WHERE a.cluster_id = ?
            """, (cluster_id,))

            # Check for empty DataFrame
            if txs.empty: return

            # Drop rows with NaN addresses
            txs.dropna(subset=['from_addr', 'to_addr'], inplace=True)
            if txs.empty: return # Check again after dropping NaNs

            G = nx.from_pandas_edgelist(txs, 'from_addr', 'to_addr', ['value_eth'], create_using=nx.DiGraph())
            plt.figure(figsize=(18, 18))
            pos = nx.spring_layout(G, k=0.15, iterations=50)
            node_sizes = [G.degree(n) * 100 + 50 for n in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True)
            plt.title(f'Network Graph for Cluster ID: {cluster_id}', size=20)
            plt.axis('off')
            
            output_path = self.output_dir / f"cluster_network_graph_{cluster_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved static cluster network graph to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save static network for cluster {cluster_id}: {e}")

    def plot_risk_contribution_bar_chart(self, address: str):
        """Creates a bar chart showing the breakdown of an address's final risk score."""
        # This method remains as previously defined...
        logger.info(f"Generating risk contribution chart for address {address[:10]}...")
        try:
            components = self.database.fetch_df("""
                SELECT component_type, risk_score FROM risk_components
                WHERE address = ? AND is_active = TRUE
            """, (address,))

            if components.empty: return

            components = components.sort_values('risk_score', ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(components['component_type'], components['risk_score'], color='orangered')
            plt.xlabel('Component Risk Score')
            plt.title(f'Risk Score Contribution for Address:\n{address}')
            plt.xlim(0, 1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            output_path = self.output_dir / f"risk_contribution_{address}.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved risk contribution chart to {output_path}")
        except Exception as e:
            logger.error(f"Failed to plot risk contribution for {address}: {e}")

    def plot_multi_hop_sankey(self, path_id: str) -> go.Figure:
        """Visualizes a single suspicious multi-hop path using a Sankey diagram."""
        # This method remains as previously defined...
        logger.info(f"Generating Sankey diagram for path {path_id}...")
        try:
            path_data = self.database.fetch_one("SELECT addresses, transactions FROM suspicious_paths WHERE path_id = ?", (path_id,))
            if not path_data: return

            addresses = json.loads(path_data['addresses'])
            tx_hashes = json.loads(path_data['transactions'])

            tx_df = self.database.fetch_df(f"""
                SELECT from_addr, to_addr, value_eth FROM transactions
                WHERE hash IN ({','.join(['?']*len(tx_hashes))})
            """, tuple(tx_hashes))

            labels = list(pd.unique(addresses))
            addr_to_idx = {addr: i for i, addr in enumerate(labels)}

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, label=[f"{addr[:6]}...{addr[-4:]}" for addr in labels]),
                link=dict(
                    source=[addr_to_idx.get(row['from_addr']) for _, row in tx_df.iterrows()],
                    target=[addr_to_idx.get(row['to_addr']) for _, row in tx_df.iterrows()],
                    value=[row['value_eth'] for _, row in tx_df.iterrows()]
                ))])
            fig.update_layout(title_text=f"Multi-Hop Flow for Path ID: {path_id}", font_size=12)
            
            output_path = self.output_dir / f"multihop_sankey_{path_id}.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved multi-hop Sankey diagram to {output_path}")

            return fig
        
        except Exception as e:
            logger.error(f"Failed to create Sankey diagram for path {path_id}: {e}")
            return go.Figure()

    def plot_cluster_activity_timeline(self, num_clusters: int = 15):
        """Creates a Gantt-style chart showing activity periods of suspicious clusters."""
        # This method remains as previously defined...
        logger.info(f"Generating activity timeline for top {num_clusters} clusters...")
        try:
            clusters = self.database.fetch_df(f"""
                SELECT a.cluster_id, COUNT(DISTINCT a.address) as size, MIN(t.timestamp) as first_tx, MAX(t.timestamp) as last_tx
                FROM addresses a JOIN transactions t ON (a.address = t.from_addr OR a.address = t.to_addr)
                WHERE a.cluster_id IS NOT NULL GROUP BY a.cluster_id ORDER BY size DESC LIMIT ?
            """, (num_clusters,))

            if clusters.empty: return

            clusters['first_tx'] = pd.to_datetime(clusters['first_tx'], unit='s')
            clusters['last_tx'] = pd.to_datetime(clusters['last_tx'], unit='s')
            clusters['duration'] = clusters['last_tx'] - clusters['first_tx']
            clusters = clusters.sort_values('first_tx')

            fig, ax = plt.subplots(figsize=(15, 8))
            y_labels = [f"Cluster {cid} (Size: {size})" for cid, size in zip(clusters['cluster_id'], clusters['size'])]
            ax.barh(y_labels, clusters['duration'], left=clusters['first_tx'], color='teal', alpha=0.7)
            plt.title(f'Activity Timeline for Top {num_clusters} Largest Clusters', size=16)
            
            output_path = self.output_dir / "cluster_activity_timeline.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved cluster activity timeline to {output_path}")
        except Exception as e:
            logger.error(f"Failed to plot cluster timeline: {e}")

    # +++ NEW VISUALIZATION 1: ATTRIBUTION GRAPH +++
    def plot_attribution_graph(self):
        """
        Generates an interactive network graph of addresses linked by behavioral similarity.
        """
        logger.info("Generating attribution graph from behavioral analysis...")
        try:
            links_df = self.database.fetch_df("""
                SELECT source_address, target_address, similarity_score
                FROM attribution_links
                WHERE similarity_score > 0.8
            """)

            if links_df.empty:
                logger.warning("No high-confidence attribution links found to visualize.")
                return

            G = nx.from_pandas_edgelist(links_df, 'source_address', 'target_address', ['similarity_score'])
            pos = nx.spring_layout(G, k=0.3, iterations=50)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.7, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x, node_y, node_text, node_size = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                degree = G.degree(node)
                node_size.append(10 + degree * 3)
                node_text.append(f"Address: {node}<br>Connections: {degree}")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlOrRd',
                    size=node_size,
                    colorbar=dict(thickness=15, title='Node Connections'),
                    line_width=2))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title_text='Address Attribution Network (Behavioral Similarity)',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            template="plotly_dark"
                        ))
            
            output_path = self.output_dir / "attribution_network.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved attribution network graph to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot attribution graph: {e}", exc_info=True)

    # +++ NEW VISUALIZATION 2: FREQUENT PATTERNS (TTPs) +++
    def plot_frequent_patterns_barchart(self):
        """
        Generates a bar chart of the most frequent behavioral patterns (TTPs).
        """
        logger.info("Generating frequent behavioral patterns (TTPs) chart...")
        try:
            patterns_df = self.database.fetch_df("""
                SELECT sequence, support, pattern_type
                FROM frequent_behavioral_patterns
                ORDER BY support DESC
                LIMIT 20
            """)

            if patterns_df.empty:
                logger.warning("No frequent behavioral patterns found to visualize.")
                return

            fig = px.bar(
                patterns_df,
                x='support',
                y='sequence',
                orientation='h',
                color='pattern_type',
                title='Top 20 Frequent Behavioral Patterns (TTPs)',
                labels={'support': 'Support (Frequency)', 'sequence': 'Behavioral Sequence'},
                template='plotly_dark'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            
            output_path = self.output_dir / "frequent_behavioral_patterns.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved frequent patterns chart to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot frequent patterns chart: {e}", exc_info=True)

    def plot_temporal_anomaly_chart(self):
        """
        Plots transaction volume over time and highlights periods of
        anomalous activity detected by the DynamicTemporalNetworkAnalyzer.
        """
        logger.info("Generating temporal anomaly chart...")
        try:
            # 1. Get overall transaction activity over time
            tx_activity_df = self.database.fetch_df("""
                SELECT
                    strftime(to_timestamp(timestamp), '%Y-%m-%d') as activity_date,
                    COUNT(*) as transaction_count,
                    SUM(value_eth) as total_volume
                FROM transactions
                GROUP BY activity_date
                ORDER BY activity_date;
            """)

            if tx_activity_df.empty:
                logger.warning("No transaction activity found for temporal anomaly chart.")
                return

            # 2. Get detected temporal anomalies
            anomaly_df = self.database.fetch_df("""
                SELECT results_json FROM advanced_analysis_results
                WHERE analysis_type = 'dynamic_temporal_network'
            """)
            
            anomalies = []
            if not anomaly_df.empty:
                results_json = json.loads(anomaly_df.iloc[0]['results_json'])
                suspicious_patterns = results_json.get('top_suspicious_patterns', [])
                for pattern in suspicious_patterns:
                    if pattern.get('start_time'):
                        anomalies.append(datetime.fromtimestamp(pattern['start_time']).strftime('%Y-%m-%d'))

            # 3. Create the plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=tx_activity_df['activity_date'], y=tx_activity_df['transaction_count'], name='Transaction Count', marker_color='lightblue'), secondary_y=False)
            fig.add_trace(go.Scatter(x=tx_activity_df['activity_date'], y=tx_activity_df['total_volume'], name='Transaction Volume (ETH)', line=dict(color='orange')), secondary_y=True)

            # Add anomaly highlights
            for anomaly_date in set(anomalies):
                fig.add_vrect(
                    x0=anomaly_date, x1=(pd.to_datetime(anomaly_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    fillcolor="red", opacity=0.2, line_width=0,
                    annotation_text="Anomaly", annotation_position="top left"
                )

            fig.update_layout(title_text="Transaction Activity Over Time with Anomaly Detection", template="plotly_dark", xaxis_title="Date", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
            fig.update_yaxes(title_text="Volume (ETH)", secondary_y=True)

            output_path = self.output_dir / "temporal_anomaly_activity.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved temporal anomaly chart to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot temporal anomaly chart: {e}", exc_info=True)

    ### =========================================================== ###
    ### VISUALIZATIONS ADAPTED FROM pattern_cluster_visualizer.py   ###
    ### =========================================================== ###
    
    def plot_behavioral_dashboard(self, top_k: int = 5)-> go.Figure:
        """
        Creates an overview dashboard of top behavioral pattern clusters.
        Adapted from your original pattern_cluster_visualizer.py script.
        """
        logger.info(f"Generating behavioral pattern dashboard for top {top_k} clusters...")
        try:
            # Data for this plot comes from deposit_withdrawal_patterns
            query = f"""
            SELECT 
                p.pattern_cluster_id,
                p.pattern_risk_score,
                p.interleaving_score,
                p.time_consistency_score
            FROM deposit_withdrawal_patterns p
            WHERE p.pattern_cluster_id IS NOT NULL
            """
            df = self.database.fetch_df(query)
            if df.empty:
                logger.warning("No behavioral pattern data found for dashboard.")
                return

            # Summarize by cluster
            summary = df.groupby('pattern_cluster_id').agg(
                size=('pattern_cluster_id', 'count'),
                avg_risk=('pattern_risk_score', 'mean'),
                avg_interleaving=('interleaving_score', 'mean')
            ).reset_index().nlargest(top_k, 'size')

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cluster Sizes', 'Risk vs. Interleaving'),
                specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
            )

            # Bar chart of cluster sizes
            fig.add_trace(go.Bar(
                x=[f"C{c}" for c in summary['pattern_cluster_id']],
                y=summary['size'],
                marker_color=self.color_palette
            ), row=1, col=1)

            # Scatter plot of risk vs. interleaving
            fig.add_trace(go.Scatter(
                x=summary['avg_interleaving'],
                y=summary['avg_risk'],
                mode='markers',
                marker=dict(size=summary['size'], sizemin=4, sizemode='area', color=summary['avg_risk'], colorscale='Reds'),
                text=[f"C{c}" for c in summary['pattern_cluster_id']]
            ), row=1, col=2)
            
            fig.update_layout(title_text="Behavioral Pattern Cluster Dashboard", showlegend=False, height=500)
            fig.update_xaxes(title_text="Cluster", row=1, col=1)
            fig.update_yaxes(title_text="Address Count", row=1, col=1)
            fig.update_xaxes(title_text="Avg. Interleaving Score", row=1, col=2)
            fig.update_yaxes(title_text="Avg. Risk Score", row=1, col=2)

            output_path = self.output_dir / "behavioral_pattern_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved behavioral dashboard to {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Failed to create behavioral dashboard: {e}")
            return go.Figure()

    def plot_pattern_cluster_dashboard(self, top_k: int = 5) -> go.Figure:
        """Creates an overview dashboard of top behavioral pattern clusters."""
        logger.info(f"Generating behavioral pattern dashboard for top {top_k} clusters...")
        try:
            query = f"""
            WITH cluster_summary AS (
                SELECT
                    pattern_cluster_id,
                    COUNT(*) as size,
                    AVG(pattern_risk_score) as avg_risk,
                    AVG(interleaving_score) as avg_interleaving,
                    FIRST(pattern_type) as pattern_type -- Assumes pattern_type is consistent per cluster
                FROM deposit_withdrawal_patterns
                WHERE pattern_cluster_id IS NOT NULL
                GROUP BY pattern_cluster_id
            )
            SELECT * FROM cluster_summary ORDER BY size DESC LIMIT {top_k};
            """
            summary = self.database.fetch_df(query)
            if summary.empty:
                logger.warning("No behavioral pattern data found for dashboard.")
                return go.Figure()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cluster Sizes', 'Risk vs. Interleaving'),
                specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
            )

            fig.add_trace(go.Bar(
                x=[f"C{c}" for c in summary['pattern_cluster_id']],
                y=summary['size'],
                marker_color=self.color_palette
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=summary['avg_interleaving'],
                y=summary['avg_risk'],
                mode='markers+text',
                marker=dict(size=summary['size'], sizemin=4, sizemode='area', color=summary['avg_risk'], colorscale='Reds'),
                text=[f"C{c}" for c in summary['pattern_cluster_id']]
            ), row=1, col=2)
            
            fig.update_layout(title_text="Behavioral Pattern Cluster Dashboard", showlegend=False, height=500, template="plotly_dark")
            fig.update_xaxes(title_text="Cluster", row=1, col=1)
            fig.update_yaxes(title_text="Address Count", row=1, col=1)
            fig.update_xaxes(title_text="Avg. Interleaving Score", row=1, col=2)
            fig.update_yaxes(title_text="Avg. Risk Score", row=1, col=2)

            output_path = self.output_dir / "behavioral_pattern_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved behavioral dashboard to {output_path}")
            return fig

        except Exception as e:
            logger.error(f"Failed to create behavioral dashboard: {e}", exc_info=True)
            return go.Figure()

    def plot_pattern_similarity_heatmap(self, top_k: int = 10) -> go.Figure:
        """Creates a similarity heatmap between the top_k behavioral clusters."""
        logger.info(f"Generating behavioral pattern similarity heatmap for top {top_k} clusters...")
        try:
            # Get top clusters by size
            top_clusters_df = self.database.fetch_df(f"""
                SELECT pattern_cluster_id FROM deposit_withdrawal_patterns
                WHERE pattern_cluster_id IS NOT NULL
                GROUP BY pattern_cluster_id ORDER BY COUNT(*) DESC LIMIT {top_k}
            """)
            if top_clusters_df.empty:
                logger.warning("Not enough clusters to generate similarity heatmap.")
                return go.Figure()
            
            top_clusters = top_clusters_df['pattern_cluster_id'].tolist()
            placeholders = ','.join(['?'] * len(top_clusters))

            # Fetch features for these clusters
            query = f"SELECT * FROM deposit_withdrawal_patterns WHERE pattern_cluster_id IN ({placeholders})"
            features_df = self.database.fetch_df(query, tuple(top_clusters))

            # Calculate centroids
            feature_cols = features_df.select_dtypes(include=np.number).columns.drop(['pattern_cluster_id', 'pattern_risk_score'], errors='ignore')
            centroids = features_df.groupby('pattern_cluster_id')[feature_cols].mean()
            
            # Calculate cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(centroids)

            cluster_labels = [f"C{c}" for c in centroids.index]
            fig = px.imshow(
                similarity_matrix,
                x=cluster_labels,
                y=cluster_labels,
                labels=dict(color="Similarity"),
                title="Behavioral Pattern Cluster Similarity (Cosine)",
                color_continuous_scale='Blues'
            )
            
            output_path = self.output_dir / "behavioral_pattern_similarity.png"
            fig.write_image(output_path, width=800, height=700, scale=2)
            logger.info(f"Saved behavioral similarity heatmap to {output_path}")
            return fig

        except Exception as e:
            logger.error(f"Failed to create similarity heatmap: {e}", exc_info=True)
            return go.Figure()