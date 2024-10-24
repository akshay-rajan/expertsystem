function handleFileChange(event) {
  const file = event.target.files[0];
  if (file) $('#upload-btn').prop('disabled', false);
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
      // Append tick icon (removes upload field)
      fileInput.parentElement.innerHTML = file.name + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
      $('#hyperparameter-div').removeClass('d-none');
    })
    .catch(error => {
      // Reactivate file input field
      fileInput.disabled = false;
      // Alert user and reload page
      alert('An error occurred while uploading the file. Please try again.');
      location.reload();
      console.error('Could not store file: ', error);
    });
  }
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
    div.classList.add('form-check-inline');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.classList.add('form-check-input');
    checkbox.name = 'features';
    checkbox.value = column;
    checkbox.id = `feature-${column}`;
    const label = document.createElement('label');
    label.className = 'form-check-label';
    label.htmlFor = `feature-${column}`;
    label.textContent = column;
    div.appendChild(checkbox);
    div.appendChild(label);
    featuresDiv.appendChild(div);
  });
}

function validateForm() {
  const features = document.querySelectorAll('input[name="features"]:checked');
  const hyperparameters = $('.hyperparameter');
  const alert = $('#alert');
  // Check if at least one feature is selected
  if (features.length === 0) {
    alert.text('Please select at least one feature.');
    alert.removeClass('d-none');
    return false;
  }
  // Validate each hyperparameter
  for (let i = 0; i < hyperparameters.length; i++) {
    const value = hyperparameters[i].value;
    // Validate Select dropdown
    if (hyperparameters[i].tagName.toLowerCase() === 'select') {
      if (!value) {
        alert.text(`Please select a value for ${hyperparameters[i].name}.`);
        alert.removeClass('d-none');
        return false;
      }
    // Validate text input (number)
    } else if (isNaN(value) || value <= 0) {
      alert.text(`Please enter a valid value for ${hyperparameters[i].name}.`);
      alert.removeClass('d-none');
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

function activateBuildButton() {
  $('#build-btn-div1').removeClass('d-none');
  $('#build-btn-div2').addClass('d-none');
}
