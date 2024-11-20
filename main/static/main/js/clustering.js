document.getElementById('generate-plot-btn').addEventListener('click', () => {
  const xAxisIndex = document.getElementById('x-axis-select').value;
  const yAxisIndex = document.getElementById('y-axis-select').value;

  fetch('/get_cluster_plot/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCSRFToken()
    },
    body: JSON.stringify({ x_axis: xAxisIndex, y_axis: yAxisIndex })
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('Received plot data:', data);  
      console.log(data['data']);
      const plotDiv = document.getElementById('plotly-plot');
      plotDiv.innerHTML = ''; // Clear existing plot
      try {
        // Render the plot using Plotly
        Plotly.newPlot(plotDiv, data);
        console.log('Plot rendered successfully');
      } catch (err) {
        console.error('Error rendering plot:', err);
        showError("Error", 'An error occurred while rendering the plot.');
      }
    })
    .catch(error => {
      console.error('Error fetching the plot:', error);
      showWarningToast('Error fetching the plot');
    });
});
  