function handleFileChange(event) {
  const file = event.target.files[0];
  if (file) $('#upload-btn').prop('disabled', false);
}

function handlePreloadedFileChange(event) {
  const dataset = event.target.value;
  if (dataset) $('#upload-btn').prop('disabled', false);
}

function handleDataset(event) {
  $('.dataset-selection').addClass('d-none');

  if ($('#option-upload').is(':checked')) {
    handleFileUpload(event);
  } else {
    handleFileSelection(event);
  }
}

function handleFileUpload(event) {
  event.preventDefault();
  const fileInput = document.getElementById('dataset');
  const file = fileInput.files[0];

  if (file) {
    $('#upload-btn').addClass('d-none');
    $('#build-btn').removeClass('d-none');

    // Save file to the server
    const formData = new FormData();
    formData.append('file', file);

    fetch('/save_file/', {
      method: 'POST',
      body: formData,
      headers: {
        'X-CSRFToken': getCSRFToken()
      }
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      fileInput.disabled = true;

      // Fetch formatted data from the server after file upload is successful
      return fetch('/get_file/', {
        method: 'POST',
        headers: {
          'X-CSRFToken': getCSRFToken()
        }
      });
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      // Populate Checkboxes
      populateFeatureCheckboxes(data.columns);

      // Plot heatmap with correlation matrix
      const correlationMatrix = data.correlation_matrix;
      plotHeatMap(formatCorrelationMatrix(correlationMatrix));
      plotlyHeatMap(data.plot);
      plotScatter(data.scatter, data.columns);

      // Append tick icon (removes upload field)
      const fileParent = fileInput.parentElement;
      fileParent.innerHTML = file.name + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
      fileParent.classList.add('mb-3');

      // Display additional configuration options
      $('#hyperparameter-div').removeClass('d-none');
      $('.optional-div').removeClass('d-none');
    })
    .catch(error => {
      // Reactivate file input field
      fileInput.disabled = false;
      // Alert user and reload page
      showError('Upload Error!', 'An error occurred while uploading the file. Please try again.');
      console.error('Could not store file: ', error);
    });
  }
}

function handleFileSelection(event) {
  event.preventDefault();

  const dataset = document.getElementById('preloaded-dataset').value;
  if (!dataset) {
    showWarningToast('Please select a preloaded dataset.');
    return;
  }

  $('#upload-btn').addClass('d-none');
  $('#build-btn').removeClass('d-none');

  // Send the selected preloaded dataset to the server
  const formData = new FormData();
  formData.append('preloaded_dataset', dataset);

  fetch('/save_file/', {
    method: 'POST',
    body: formData,
    headers: {
      'X-CSRFToken': getCSRFToken()
    }
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // Fetch the formatted data for the preloaded dataset
    return fetch('/get_file/', {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCSRFToken()
      }
    });
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // Populate feature selection
    populateFeatureCheckboxes(data.columns);

    // Plot heatmap with the correlation matrix
    const correlationMatrix = data.correlation_matrix;
    plotHeatMap(formatCorrelationMatrix(correlationMatrix));
    plotlyHeatMap(data.plot);
    plotScatter(data.scatter, data.columns);

    // Display the preloaded dataset name with a tick icon
    const preloadedDiv = document.getElementById('preloaded-div');
    preloadedDiv.innerHTML = dataset + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
    preloadedDiv.classList.add('mb-3');

    // Display additional configuration options
    $('#hyperparameter-div').removeClass('d-none');
    $('.optional-div').removeClass('d-none');
  })
  .catch(error => {
    // Handle errors during dataset retrieval
    showError('Selection Error!', 'An error occurred while retrieving the dataset. Please try again.');
    console.error('Error during dataset retrieval: ', error);
  });
}

function getCSRFToken() {
  const cookieValue = document.cookie.match('(^|;)\\s*csrftoken\\s*=\\s*([^;]+)');
  return cookieValue ? cookieValue.pop() : '';
}

function populateFeatureCheckboxes(columns) {
  const featuresParent = document.getElementById('features-div');
  featuresParent.classList.remove('d-none');

  const featuresDiv = document.getElementById('features');
  featuresDiv.innerHTML = '';

  columns.forEach(column => {
    const div = document.createElement('div');
    div.classList.add('form-check', 'd-inline-flex', 'me-2');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.classList.add('form-check-input');
    checkbox.name = 'features';
    checkbox.value = column.trim();
    checkbox.id = `feature-${column.trim()}`;
    const label = document.createElement('label');
    label.className = 'form-check-label';
    label.htmlFor = `feature-${column.trim()}`;
    label.textContent = column.trim();
    div.appendChild(checkbox);
    div.appendChild(label);
    featuresDiv.appendChild(div);
  });
}

function validateForm() {
  const features = document.querySelectorAll('input[name="features"]:checked');
  const hyperparameters = $('.hyperparameter');
  // Check if at least one feature is selected
  if (features.length === 0) {
    showWarningToast('Please select at least one feature.');
    return false;
  }
  // Validate each hyperparameter
  for (let i = 0; i < hyperparameters.length; i++) {
    const value = hyperparameters[i].value;
    // Validate Select dropdown
    if (hyperparameters[i].tagName.toLowerCase() === 'select') {
      if (!value) {
        showWarningToast(`Please select a value for ${hyperparameters[i].name}.`);
        return false;
      }
    // Validate text input (number)
    } else if (isNaN(value) || value <= 0) {
      showWarningToast(`Please enter a valid value for ${hyperparameters[i].name}.`);
      return false;
    }
  }
  displayLoader();
  return true;
}

function displayLoader() {
  $('.page').addClass('d-none');
  $('.loader').addClass('d-flex');
}

// ! Heatmap
function plotlyHeatMap(data) {
  Plotly.newPlot('plotly-heatmap', JSON.parse(data));
  activateBuildButton();
}
function plotHeatMap(data) {
  // Remove any existing SVG elements
  d3.select("#canvas-1").selectAll("*").remove();

  // Set the dimensions and margins of the graph
  var margin = {top: 80, right: 25, bottom: 80, left: 100},
      width = 450 - margin.left - margin.right,
      height = 450 - margin.top - margin.bottom;

  // Append the svg object to the body of the page
  var svg = d3.select("#canvas-1")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  // Extract unique groups and variables from the data
  var myGroups = [...new Set(data.map(d => d.group))];
  var myVars = [...new Set(data.map(d => d.variable))];

  // Build X scales and axis
  var x = d3.scaleBand()
  .range([ 0, width ])
  .domain(myGroups)
  .padding(0.05);
  svg.append("g")
  .style("font-size", 12)
  .attr("transform", "translate(0,0)") // Adjusted to start from the top
  .call(d3.axisTop(x).tickSize(0)) // Changed to axisTop
  .selectAll("text")
    .attr("transform", "rotate(-90)")
    .style("text-anchor", "start")
    .attr("dx", ".8em")
    .attr("dy", "-.15em");

  // Hide the X axis line and ticks
  svg.selectAll(".domain").style("stroke", "none");
  svg.selectAll(".tick line").style("stroke", "none");

  // Build Y scales and axis
  var y = d3.scaleBand()
  .range([ 0, height ]) // Adjusted to start from the top
  .domain(myVars)
  .padding(0.05);
  svg.append("g")
  .style("font-size", 12)
  .call(d3.axisLeft(y).tickSize(0))
  .selectAll("text")
    .style("text-anchor", "end")
    .attr("dx", "-.8em")
    .attr("dy", ".15em");

  // Hide the Y axis line and ticks
  svg.selectAll(".domain").style("stroke", "none");
  svg.selectAll(".tick line").style("stroke", "none");

  // Build color scale
  var myColor = d3.scaleSequential()
    .interpolator(d3.interpolateGnBu)
    .domain([-1, 1]);


  // Add the squares
  svg.selectAll()
    .data(data, function(d) {return d.group+':'+d.variable;})
    .enter()
    .append("rect")
      .attr("x", function(d) { return x(d.group); })
      .attr("y", function(d) { return y(d.variable); })
      .attr("rx", 4)
      .attr("ry", 4)
      .attr("width", x.bandwidth())
      .attr("height", y.bandwidth())
      .style("fill", function(d) { return myColor(d.value); })
      .style("stroke-width", 4)
      .style("stroke", "none")
      .style("opacity", 0.8)
  
  // Add text to the squares
  svg.selectAll()
  .data(data, function(d) { return d.group + ':' + d.variable; })
  .enter()
  .append("text")
    .attr("x", function(d) { return x(d.group) + x.bandwidth() / 2; })
    .attr("y", function(d) { return y(d.variable) + y.bandwidth() / 2; })
    .attr("dy", ".35em")
    .attr("text-anchor", "middle")
    .style("fill", "black") // Adjust text color based on background color for better visibility
    .style("font-size", "10px")
    .text(function(d) { return d.value.toFixed(2); });

  // Add title to graph
  svg.append("text")
    .attr("x", 0)
    .attr("y", height + margin.bottom / 3) // Position below the heatmap
    .attr("text-anchor", "left")
    .style("font-size", "20px")
    .text("Heatmap");
  
  // Add subtitle to graph
  svg.append("text")
    .attr("x", 0)
    .attr("y", height + margin.bottom / 3 + 30) // Position below the title
    .attr("text-anchor", "left")
    .style("font-size", "12px")
    .style("fill", "grey")
    .style("max-width", 400)
    .text("Correlation between each pair of features.");

    activateBuildButton();
}
function toggleHeatmaps() {
  document.querySelectorAll('.heatmaps').forEach(heatmap => heatmap.classList.toggle('d-none'));
}

// ! Scatter Plot
function plotScatter(data, columns) {
  if (columns.length < 2) return;
  $('#scatter-container').removeClass('d-none');
  $('#plotly-scatter').html('');

  // Populate X and Y axis dropdowns
  const xSelect = document.getElementById('x-axis-select');
  const ySelect = document.getElementById('y-axis-select');
  xSelect.innerHTML = '';
  ySelect.innerHTML = '';
  columns.forEach(column => {
    const option = document.createElement('option');
    option.value = column;
    option.textContent = column;
    xSelect.appendChild(option.cloneNode(true));
    ySelect.appendChild(option.cloneNode(true));
  });
  // Set default values for X and Y axes
  xSelect.value = columns[0];
  ySelect.value = columns[1];

  Plotly.newPlot('plotly-scatter', JSON.parse(data));
}
function generateScatter() {
  // Get user-selected x and y axes
  const xAxis = document.getElementById('x-axis-select').value;
  const yAxis = document.getElementById('y-axis-select').value;

  if (!xAxis || !yAxis) {
    showWarningToast('Please select both X and Y axes.');
    return;
  }

  fetch('/get_scatter_plot/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCSRFToken()
    },
    body: JSON.stringify({ x_axis: xAxis, y_axis: yAxis })
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      try {
        $('#plotly-scatter').html('');
        Plotly.newPlot('plotly-scatter', JSON.parse(data));
      } catch (err) {
        console.error('Error rendering scatter plot:', err);
        showWarningToast('Error fetching the scatter plot!');
      }
    })
    .catch(error => {
      console.error('Error fetching scatter plot:', error);
      showWarningToast('Error fetching the scatter plot!');
    });
}

function formatCorrelationMatrix(matrix) {
  let formattedData = [];
  for (let group in matrix) {
    for (let variable in matrix[group]) {
      formattedData.push({
        group: group,
        variable: variable,
        value: matrix[group][variable]
      });
    }
  }
  return formattedData;
}

function activateBuildButton() {
  $('#build-btn-div1').removeClass('d-none');
  $('#build-btn-div2').addClass('d-none');
}
