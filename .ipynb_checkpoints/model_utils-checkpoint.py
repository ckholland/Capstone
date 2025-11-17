import joblib

def load_ols():
    ols = joblib.load("ols.joblib")
    return ols

def load_ridge():
	ridge = joblib.load("ridge.joblib")
	return ridge
	
def load_lasso():
    lasso = joblib.load("lasso.joblib")
	return lasso
