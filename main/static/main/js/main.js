document.addEventListener('DOMContentLoaded', () => {
  $('#predictionForm').slideDown();
  setTimeout(() => {
  }, 2000);
});
// ! Make prediction using the saved model
async function makePrediction(event) {
  event.preventDefault(); 

  const predictionResult = document.getElementById('prediction-result');
  predictionResult.className = 'alert m-2';
  // Show loading spinner
  predictionResult.innerHTML = `<div class="spinner-border text-info" role="status"><span class="sr-only">Loading...</span></div>`;
  
  // Read input values from the form and convert them to JSON 
  const formData = new FormData(event.target);
  let inputData = {};
  formData.forEach((value, key) => {
    if (key === 'model_path' || key == 'target') return;
    inputData[key] = Number(value);
  });
  const modelPath = formData.get('model_path');
  const target = formData.get('target');

  inputData = Object.values(inputData);

  // Make a POST request to '/predict' to make the prediction
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input: inputData, model_path: modelPath })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Unknown error occurred');
    }

    const data = await response.json();
    predictionResult.classList.add('alert-success');
    predictionResult.innerHTML = target + ": " + data.predictions;
  } catch (error) {
    predictionResult.classList.add('alert-danger');
    predictionResult.innerHTML = error.message;
  }
}
