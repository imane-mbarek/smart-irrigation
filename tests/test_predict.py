import pytest
from src.predict import predict

def test_predict_full_cycle():
    """Vérifie que la fonction predict renvoie une réponse complète."""
    # Simulation d'un cas de besoin d'eau
    result = predict(
        soil_moisture=10.0,
        temperature=40.0,
        humidity=15.0,
        rainfall=0.0,
        soil_type="Sandy",
        crop_type="Cotton"
    )

    # Vérification des clés de sortie (Dictionnaire)
    assert isinstance(result, dict)
    assert "irrigate" in result
    assert "statut_pompe" in result

    # Si irrigate=1, on vérifie que la régression a calculé des litres
    if result["irrigate"] == 1:
        assert result["litres_eau"] > 0
        assert result["statut_pompe"] == "ON"