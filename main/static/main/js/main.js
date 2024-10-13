async function makePrediction() {
  // Get the input data from the form
  
  console.log('Input data:', inputData);
  // try {
  //   const response = await fetch('/predict/', {
  //     method: 'POST',
  //     headers: {
  //         'Content-Type': 'application/json'
  //     },
  //     body: JSON.stringify({ input: inputData })
  //   });

  //   if (!response.ok) {
  //     throw new Error('Network response was not ok');
  //   }

  //   const data = await response.json();
  //   console.log('Predictions:', data.predictions);
  // } catch (error) {
  //   console.error('Error:', error);
  // }
}