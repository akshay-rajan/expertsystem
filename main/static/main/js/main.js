document.addEventListener('DOMContentLoaded', () => {
  $('.loader').addClass('d-none'); 
});

// Get the CSRF token from the cookie
function getCSRFToken() {
  const cookies = document.cookie.split(';');
  for (let i = 0; i < cookies.length; i++) {
    const cookie = cookies[i].trim();
    if (cookie.startsWith('csrftoken=')) {
      return cookie.substring('csrftoken='.length, cookie.length);
    }
  }
  return '';
}

// ! Make prediction using the saved model
async function makePrediction(event) {
  event.preventDefault(); 

  const predictionResult = document.getElementById('prediction-result');
  predictionResult.className = 'alert m-2';
  // Show loading spinner
  predictionResult.innerHTML = `<div class="spinner-border text-info" role="status"><span class="visually-hidden">Loading...</span></div>`;
  
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
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken(),
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
    
    // If the user is in the platform tour, continue the tour
    predTour();
  } catch (error) {
    predictionResult.classList.add('alert-danger');
    predictionResult.innerHTML = error.message;
  }
}
function getCSRFToken() {
  const cookies = document.cookie.split(';');
  for (let i = 0; i < cookies.length; i++) {
    const cookie = cookies[i].trim();
    if (cookie.startsWith('csrftoken=')) {
      return cookie.substring('csrftoken='.length, cookie.length);
    }
  }
  return '';
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
      confirmButton: 'copy-code-btn btn btn-primary d-flex justify-content-center align-items-center',
      cancelButton: 'cancel-code-btn btn btn-danger d-flex justify-content-center align-items-center'
    },
    showCloseButton: true,
    showConfirmButton: true,
    showCancelButton: true,
    confirmButtonText: `
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" class="bi bi-copy" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1z"/>
      </svg>`,
    cancelButtonText: `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-x-circle" viewBox="0 0 16 16">
        <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>
        <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708"/>
      </svg>`,
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
      showInfoToast('Code copied to clipboard!');
    }
  });
});

// ! Gauges for Evaluation Metrics
function colorPicker(value, a, b, c) {
  if (value <= a) {
    return "#ef4655"; // Red for low range
  } else if (value <= b) {
    return "#f7aa38"; // Orange for mid-low range
  } else if (value <= c) {
    return "#fffa50"; // Yellow for mid-high range
  } else {
    return "#00bfa5"; // Green for high range
  }
}
function initializeGauge(elementId, originalValue) {
  const gaugeConfig = {
    max: 100,
    dialStartAngle: -90,
    dialEndAngle: -90.001,
    color: function(value) {
      // Use rangeColorPicker with specific ranges
      return colorPicker(value, 25, 50, 75, 100);
    },
    label: function(value) {
      return `${value.toFixed(2)}%`;
    }
  };
  Gauge(document.getElementById(elementId), {
    ...gaugeConfig,
    value: originalValue,
  });
}
function reverseColorPicker(value, a, b, c) {
  if (value <= a) {
    return "#00bfa5"; // Green for low range
  } else if (value <= b) {
    return "#fffa50"; // Yellow for mid-low range
  } else if (value <= c) {
    return "#f7aa38"; // Orange for mid-high range
  } else {
    return "#ef4655"; // Red for high range
  }
}
function initializeRegressionGauge(elementId, originalValue, reversed) {
  const gaugeConfig = {
    max: 1,
    dialStartAngle: -90,
    dialEndAngle: -90.001,
    color: function(value) {
      return reversed ? reverseColorPicker(originalValue, 0.25, 0.5, 0.75) : colorPicker(originalValue, 0.25, 0.5, 0.75);
    },
    label: function() {
      // Display the original value in the label, regardless of gauge's max value
      return `${originalValue.toFixed(4)}`;
    },
  };
  // Set the gauge value to capped max if originalValue exceeds max
  const displayValue = Math.min(originalValue, gaugeConfig.max);
  Gauge(document.getElementById(elementId), {
    ...gaugeConfig,
    value: 1 - displayValue
  });
}

// ? Toasts and Alerts
function showNotification(context) {
  // { title, message, confirmText, cancelText, onConfirm, onDismiss, actionArgs }
  Swal.fire({
    title: context.title,
    text: context.message,
    confirmButtonText: context.confirmText,
    showCancelButton: context.cancelText ? true : false,
    cancelButtonText: context.cancelText,
    showCloseButton: true,
    toast: true,
    position: 'bottom-right',
  }).then((result) => {
    if (result.isConfirmed && context.onConfirm) {
      context.actionArgs ? context.onConfirm(...context.actionArgs) : context.onConfirm();
    }
    if (result.dismiss && context.onDismiss) {
      context.onDismiss();
    }
  });
}
function showWarningToast(message) {
  Toastify({
    text: message,
    duration: 2500,
    gravity: "bottom", // position at the top of the page
    position: "right", // right side of the page
    close: false,
    stopOnFocus: true,
    pauseOnHover: true,
    hideProgressBar: false,
    style: {
      background: "linear-gradient(to right, #E66A1E, #F60015)",
      color: "#fff",
    }
  }).showToast();
}

function showError(title, message=null, func=null, args=null) {
  Swal.fire({
    title: title,
    html: message ? message : 'An error occurred. Please try again.',
    icon: 'error',
    confirmButtonText: 'Close',
    customClass: {
      confirmButton: 'btn btn-primary'
    }
  }).then(() => {
    if (func) {
      console.log(...args);
      args ? func(...args) : func();
    }
    location.reload();
  });
}

function showInfoToast(message) {
  Toastify({
    text: message,
    duration: 2500,
    gravity: "bottom", 
    position: "right", 
    stopOnFocus: true,
    pauseOnHover: true,
    style: {
      background: "white",
      color: "#0096ff",
    }
  }).showToast();
}

document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname; // Get the current path
  const navLinks = document.querySelectorAll('.nav-link');

  navLinks.forEach(link => {
    if (link.getAttribute('href') === path) {
      link.parentElement.classList.add('active'); // Add active class to the current link
    }
  });
});

// Dataset upload method selection
function handleDatasetOptionChange(event) {
  const selectedOption = event.target.value;
  const uploadDiv = document.getElementById('upload-div');
  const preloadedDiv = document.getElementById('preloaded-div');
  const uploadBtnDiv = $('.dataset-selection-upload');
  const preloadBtnDiv = $('.dataset-selection-preload');

  if (selectedOption === 'upload') {
    uploadDiv.classList.remove('d-none');
    uploadBtnDiv.addClass('selected');
    preloadedDiv.classList.add('d-none');
    preloadBtnDiv.removeClass('selected');
  } else if (selectedOption === 'choose') {
    uploadDiv.classList.add('d-none');
    uploadBtnDiv.removeClass('selected');
    preloadedDiv.classList.remove('d-none');
    preloadBtnDiv.addClass('selected');
  }
}

// Animations
var animateButton = function(e) {
  e.preventDefault;
  e.target.classList.remove('animate');  
  e.target.classList.add('animate');
  setTimeout(function(){
    e.target.classList.remove('animate');
  },700);
};

var bubblyButtons = document.getElementsByClassName("bubbly-button");
for (var i = 0; i < bubblyButtons.length; i++) {
  bubblyButtons[i].addEventListener('click', animateButton, false);
}

// Platform tour continues after making a prediction in KNN
function predTour() {
  if (localStorage.getItem('prediction') !== 'start') return;
  localStorage.setItem('prediction', 'end');
  introJs().setOptions({
    steps: [
      {
        element: document.querySelector('#prediction-result'),
        intro: "Great job! The model has generated the predicted values based on the input features you provided. This is the class in which the data point you entered belongs to.",
      },
      {
        element: document.querySelector('.table-responsive'),
        intro: "The table below shows the actual and predicted values for some test data. You can compare the predicted values with the actual values to understand the model's performance.",
      },
      {
        element: document.querySelector('.download-model'),
        intro: "You can download the trained model (as a .pkl file) by clicking the button below. This saved model can be used to make predictions anytime, and it can be deployed in production systems.",
      },
      {
        element: document.querySelector('.view-code-btn'),
        intro: "Here, you can view a sample Python code snippet that demonstrates how to train a KNN model using the scikit-learn library. You can replace the example dataset with your own data to train a custom model.",
      },
      {
        intro: "That's it! You have completed the tour of the platform. 12 more algorithms are waiting for you to explore. Don't forget to check out the preprocessing and visualization tools as well. Happy learning!",
      }
    ],
  }).start();
}
