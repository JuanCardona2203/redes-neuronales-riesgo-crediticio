import torch

from src.model import CreditRiskMLP

VARIABLES = [
    'person_age', 'person_income', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
]


def cargar_modelo(ruta='models/mlp_credit_risk.pt', input_dim=6):
    model = CreditRiskMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(ruta))
    model.eval()
    return model


def predecir(model, X_escalado):
    # X_escalado: arreglo (N, 6) con las variables ya escaladas en [0, 1]
    with torch.no_grad():
        prob = model(torch.tensor(X_escalado, dtype=torch.float32)).numpy().ravel()
    pred = (prob >= 0.5).astype(int)
    return prob, pred
