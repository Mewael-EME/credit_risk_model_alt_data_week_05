from src.target_engineering import compute_rfm, cluster_rfm, label_high_risk_cluster

# Load raw data
raw_df = pd.read_csv("data/processed/processed_data.csv")  # Or wherever your input is

# Step 1: Compute RFM
rfm = compute_rfm(raw_df)

# Step 2: Cluster RFM
rfm_clustered = cluster_rfm(rfm)

# Step 3: Label high-risk
risk_labels = label_high_risk_cluster(rfm_clustered)

# Step 4: Merge with processed data
raw_df = raw_df.merge(risk_labels, on="CustomerId", how="left")
raw_df["is_high_risk"] = raw_df["is_high_risk"].fillna(0).astype(int)
