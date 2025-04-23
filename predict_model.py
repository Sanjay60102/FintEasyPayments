import xmlschema
import joblib
 
# Load your trained model
model = joblib.load('model/sgd_classifier_model.pkl')
try:
            # Predict fix from your trained model
    a="The 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.03:BIC' element is invalid - The value '' is invalid according to its datatype 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.03:BICIdentifier' - The Pattern constraint failed."
    suggestion = model.predict([a])[0]
    print("💡 Suggestion:", suggestion)
except Exception as e:
    print("⚠️ Could not predict suggestion:", e)
 
    print("-" * 60)