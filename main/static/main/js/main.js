async function makePrediction(event) {
  event.preventDefault(); 

  const predictionResult = document.getElementById('prediction-result');
  predictionResult.className = 'alert m-2';
  // Show loading spinner
  predictionResult.innerHTML = `<div class="spinner-border text-info" role="status"><span class="sr-only">Loading...</span></div>`;
  
  await new Promise(resolve => setTimeout(resolve, 1000));

  const formData = new FormData(event.target);
  const inputData = formData.get('input').split(',').map(Number);
  const modelPath = formData.get('model_path');

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
    predictionResult.innerHTML = "Target: " + data.predictions;
  } catch (error) {
    predictionResult.classList.add('alert-danger');
    predictionResult.innerHTML = "Error: " + error.message;
  }
}