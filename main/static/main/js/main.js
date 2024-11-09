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

// ! Display source code of the algorithm
$('#show-code-btn').click(() => {
  // Get the inner HTML of the source code div
  const sourceCode = document.querySelector('.source-code').innerHTML;

  Swal.fire({
    title: '<div class="text-left">Source Code</div>',
    html: `${sourceCode}`,
    customClass: {
      popup: 'swal-wide',
      confirmButton: 'btn btn-primary'
    },
    showCloseButton: true,
    showConfirmButton: true,
    confirmButtonText: 'Copy', // Set default button text to "Copy Code"
    didOpen: () => {
      Prism.highlightAll(); // Highlight syntax
    },
    preConfirm: () => {
      // Copy the code to the clipboard when confirm button is clicked
      const tempElement = document.createElement("textarea");
      tempElement.value = document.querySelector('.source-code').innerText; // Plain text version of code
      document.body.appendChild(tempElement);
      tempElement.select();
      document.execCommand("copy");
      document.body.removeChild(tempElement);
    }
  }).then((result) => {
    if (result.isConfirmed) {
      // Show success message after copying
      Swal.fire({
        title: 'Copied!',
        timer: 800,
        showConfirmButton: false
      });
    }
  });
});
