def get_patient_context(patient_id: str) -> str:
    return (
        f"Paciente {patient_id}: febre há 2 dias, tosse seca, saturação 95%, "
        "hemograma pendente, PCR pendente, sem registro recente de internação."
    )