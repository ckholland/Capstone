def ridge(df):
    coef_prev = 0.4974
    coef_internal = 1.8746
    coef_external = 3.2678
    intercept = 40.7923
    ridge_score = (
        coef_prev * df["prev_score"]
        + coef_internal * df["internal_sentiment"]
        + coef_external * df["external_sentiment"]
        + intercept
    )
    return ridge_score