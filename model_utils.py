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

def lasso(df):
    coef_prev = 0.554995
    coef_internal = 0.000000
    coef_external = 1.729473
    intercept = 29.1801
    ridge_score = (
        coef_prev * df["prev_score"]
        + coef_internal * df["internal_sentiment"]
        + coef_external * df["external_sentiment"]
        + intercept
    )
    return lasso_score

def ols(df):
    coef_prev = 0.3928
    coef_internal = 7.1732
    coef_external = 3.2819
    intercept = 60.8428
    ridge_score = (
        coef_prev * df["prev_score"]
        + coef_internal * df["internal_sentiment"]
        + coef_external * df["external_sentiment"]
        + intercept
    )
    return ols_score
