# src/visualization/web_visualizers.py
"""
Functions to generate visualizations for the web application dashboard.
"""
import logging
import pandas as pd
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

def create_sankey_for_top_clusters(db: DatabaseEngine, top_n_flows: int = 25) -> go.Figure:
    """
    Creates a Sankey diagram showing ETH flow between the largest clusters.
    """
    logger.info("Generating inter-cluster Sankey diagram for web app...")
    try:
        # This query gets transactions, maps from/to addresses to their clusters,
        # filters for inter-cluster transactions, aggregates the volume, and takes the top N flows.
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
        flows_df = db.fetch_df(query)

        if flows_df.empty:
            return create_placeholder_fig("Inter-Cluster Fund Flow", "No inter-cluster transaction data found.")

        # Prepare data for Sankey
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
            title_text="Top Inter-Cluster Fund Flows",
            font_size=12,
            template="plotly_dark"
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to create Sankey diagram: {e}")
        return create_placeholder_fig("Inter-Cluster Fund Flow", f"Error: {e}")


def create_temporal_heatmap(db: DatabaseEngine) -> go.Figure:
    """
    Creates a heatmap of transaction activity by day of the week and hour.
    """
    logger.info("Generating temporal activity heatmap for web app...")
    try:
        query = "SELECT timestamp, value_eth FROM transactions"
        tx_df = db.fetch_df(query)

        if tx_df.empty:
            return create_placeholder_fig("Temporal Activity Heatmap", "No transaction data found.")

        tx_df['datetime'] = pd.to_datetime(tx_df['timestamp'], unit='s')
        tx_df['day_of_week'] = tx_df['datetime'].dt.day_name()
        tx_df['hour'] = tx_df['datetime'].dt.hour

        heatmap_data = tx_df.pivot_table(
            index='day_of_week', columns='hour', values='value_eth', aggfunc='sum'
        ).fillna(0)

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = heatmap_data.reindex(days_order)

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Total ETH Volume"),
            title="Transaction Volume by Day and Hour",
            template="plotly_dark",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to create temporal heatmap: {e}")
        return create_placeholder_fig("Temporal Activity Heatmap", f"Error: {e}")


def create_risk_distribution_histogram(db: DatabaseEngine) -> go.Figure:
    """
    Creates a histogram showing the distribution of final risk scores.
    """
    logger.info("Generating risk score distribution histogram for web app...")
    try:
        query = "SELECT risk_score FROM addresses WHERE risk_score IS NOT NULL"
        risk_df = db.fetch_df(query)

        if risk_df.empty:
            return create_placeholder_fig("Risk Score Distribution", "No risk score data found.")

        fig = px.histogram(
            risk_df, x="risk_score", nbins=50, title="Distribution of Address Risk Scores",
            template="plotly_dark", labels={'risk_score': 'Final Risk Score'}
        )
        fig.update_layout(bargap=0.1)
        return fig

    except Exception as e:
        logger.error(f"Failed to create risk distribution histogram: {e}")
        return create_placeholder_fig("Risk Score Distribution", f"Error: {e}")


# +++ NEW WEB VISUALIZATION 1: ATTRIBUTION GRAPH +++
def create_attribution_graph(db: DatabaseEngine) -> go.Figure:
    """
    Creates an interactive network graph of addresses linked by behavioral similarity for the web app.
    """
    logger.info("Generating attribution graph for web app...")
    try:
        links_df = db.fetch_df("""
            SELECT source_address, target_address, similarity_score
            FROM attribution_links
            WHERE similarity_score > 0.8
        """)

        if links_df.empty:
            return create_placeholder_fig("Attribution Network", "No high-confidence attribution links found.")

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
                colorbar=dict(thickness=15, title='Connections'),
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
        return fig

    except Exception as e:
        logger.error(f"Failed to create attribution graph for web app: {e}")
        return create_placeholder_fig("Attribution Network", f"Error: {e}")

# +++ NEW WEB VISUALIZATION 2: FREQUENT PATTERNS (TTPs) +++
def create_frequent_patterns_barchart(db: DatabaseEngine) -> go.Figure:
    """
    Creates a bar chart of the most frequent behavioral patterns (TTPs) for the web app.
    """
    logger.info("Generating frequent patterns chart for web app...")
    try:
        patterns_df = db.fetch_df("""
            SELECT sequence, support, pattern_type
            FROM frequent_behavioral_patterns
            ORDER BY support DESC
            LIMIT 20
        """)

        if patterns_df.empty:
            return create_placeholder_fig("Frequent Behavioral Patterns", "No patterns found.")

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
        return fig

    except Exception as e:
        logger.error(f"Failed to create frequent patterns chart for web app: {e}")
        return create_placeholder_fig("Frequent Behavioral Patterns", f"Error: {e}")


# +++ NEW WEB VISUALIZATION 3: RISK VS. VALUE +++
def create_risk_vs_value_scatter(db: DatabaseEngine) -> go.Figure:
    """
    Creates a scatter plot of address risk score vs. average transaction value for the web app.
    """
    logger.info("Generating risk vs. value scatter plot for web app...")
    try:
        query = """
        SELECT
            address,
            risk_score,
            total_volume_eth / total_transaction_count AS avg_tx_value,
            is_tornado
        FROM addresses
        WHERE risk_score IS NOT NULL AND total_transaction_count > 0 AND total_volume_eth > 0;
        """
        df = db.fetch_df(query)

        if df.empty:
            return create_placeholder_fig("Risk vs. Value", "No data available for this plot.")

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
            hover_data={'address': True, 'avg_tx_value': ':.4f'},
            template="plotly_dark",
            color_discrete_map={True: 'red', False: 'cornflowerblue'},
            opacity=0.7
        )
        fig.update_layout(xaxis_title="Average Transaction Value (ETH) - Log Scale", yaxis_title="Final Risk Score")
        return fig
    except Exception as e:
        logger.error(f"Failed to create risk vs. value scatter plot for web app: {e}")
        return create_placeholder_fig("Risk vs. Value", f"Error: {e}")


def create_placeholder_fig(title: str, message: str) -> go.Figure:
    """Creates a placeholder figure for when data is unavailable."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": message, "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 16}
        }],
        template="plotly_dark"
    )
    return fig