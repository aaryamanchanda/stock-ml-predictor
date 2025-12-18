from sklearn.ensemble import RandomForestClassifier


def get_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
