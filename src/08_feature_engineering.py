from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale_and_pca(train_emb, val_emb, test_emb, hold_emb):
    scaler = StandardScaler()
    train_sc = scaler.fit_transform(train_emb)
    val_sc   = scaler.transform(val_emb)
    test_sc  = scaler.transform(test_emb)
    hold_sc  = scaler.transform(hold_emb)

    pca = PCA(n_components=0.95, random_state=42)
    train_pca = pca.fit_transform(train_sc)
    val_pca   = pca.transform(val_sc)
    test_pca  = pca.transform(test_sc)
    hold_pca  = pca.transform(hold_sc)

    return train_pca, val_pca, test_pca, hold_pca
