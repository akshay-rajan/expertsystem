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
      console.log("Fetched data!");
      console.log(JSON.stringify(data.file));
      console.log(data.columns);
      // Populate Checkboxes
      populateFeatureCheckboxes(data.columns);
      // Populate Dropdown
      populateTargetDropdown(data.columns);
      // Append tick icon (removes upload field)
      fileInput.parentElement.innerHTML = file.name + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
    })
    .catch(error => {
      // Reactivate file input field
      fileInput.disabled = false;
      // Alert user and reload page
      alert('An error occurred while uploading the file. Please try again.');
      location.reload();
      console.error('Could not store file: ', error);
    });

    const reader = new FileReader();
    reader.onload = function(e) {
      const content = e.target.result;
      console.log(content);
      const columns = extractColumns(content, file.name);
      // Enable hyperparameter input
      $('#hyperparameter-div').removeClass('d-none');
      // Parse data and plot heatmap
      const { data, correlationMatrix } = parseData(content, file.name);
      plotHeatMap(correlationMatrix);
    };
    
    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else if (file.name.endsWith('.xls') || file.name.endsWith('.xlsx')) {
      reader.readAsBinaryString(file);
    } else {
      alert('Invalid file format. Please upload a CSV or Excel file.');
    }
  }
}

function getCSRFToken() {
  const cookieValue = document.cookie.match('(^|;)\\s*csrftoken\\s*=\\s*([^;]+)');
  return cookieValue ? cookieValue.pop() : '';
}

// Activate the build button after the heatmap is plotted
function activateBuildButton() {
  $('#build-btn-div1').removeClass('d-none');
  $('#build-btn-div2').addClass('d-none');
}

function extractColumns(content, fileName) {
  let columns = [];
  if (fileName.endsWith('.csv')) {
    const lines = content.split('\n');
    if (lines.length > 0) {
      columns = lines[0].split(',');
    }
  } else if (fileName.endsWith('.xls') || fileName.endsWith('.xlsx')) {
    const workbook = XLSX.read(content, { type: 'binary' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    if (json.length > 0) {
      columns = json[0];
    }
  }
  return columns;
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

function populateTargetDropdown(columns) {
  const targetParent = document.getElementById('target-div');
  targetParent.classList.remove('d-none');

  const targetSelect = document.getElementById('target');
  targetSelect.innerHTML = '';

  // Add default "select" option
  const defaultOption = document.createElement('option');
  defaultOption.value = '';
  defaultOption.textContent = 'Select';
  defaultOption.disabled = true;
  defaultOption.selected = true;
  targetSelect.appendChild(defaultOption);

  columns.forEach(column => {
    const option = document.createElement('option');
    option.value = column
      .replace(/\n/g, '')
      .replace(/\r/g, '');
    option.textContent = column
      .replace(/\n/g, '')
      .replace(/\r/g, '');
    targetSelect.appendChild(option);
  });
}

function validateForm() {
  const features = document.querySelectorAll('input[name="features"]:checked');
  const target = document.getElementById('target').value;
  const hyperparameters = $('.hyperparameter');
  const alert = $('#alert');
  // Check if at least one feature is selected
  if (features.length === 0) {
    alert.text('Please select at least one feature.');
    alert.removeClass('d-none');
    return false;
  }
  // Check if a target is selected
  if (!target) {
    alert.text('Please select a target.');
    alert.removeClass('d-none');
    return false;
  }
  // Validate each hyperparameter if present
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

// Display loading spinner on form submission
function displayLoader() {
  $('.page').addClass('d-none');
  $('.loader').addClass('d-flex');
}

function parseData(content, fileName) {
  let data = [];
  if (fileName.endsWith('.csv')) {
    const lines = content.split('\n');
    const headers = lines[0].split(',');
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length === headers.length) {
        let obj = {};
        headers.forEach((header, index) => {
          obj[header] = parseFloat(row[index]);
        });
        data.push(obj);
      }
    }
  } else if (fileName.endsWith('.xls') || fileName.endsWith('.xlsx')) {
    const workbook = XLSX.read(content, { type: 'binary' });
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    data = XLSX.utils.sheet_to_json(worksheet, { raw: true });
  }

  // Calculate correlation matrix
  const headers = Object.keys(data[0]);
  const correlationMatrix = [];
  for (let i = 0; i < headers.length; i++) {
    for (let j = 0; j < headers.length; j++) {
      const x = headers[i];
      const y = headers[j];
      const correlation = calculateCorrelation(data, x, y);
      correlationMatrix.push({ group: x, variable: y, value: correlation });
    }
  }
  return { data, correlationMatrix };
}

function calculateCorrelation(data, x, y) {
  const xValues = data.map(d => d[x]);
  const yValues = data.map(d => d[y]);
  const n = xValues.length;
  const xMean = d3.mean(xValues);
  const yMean = d3.mean(yValues);
  const numerator = d3.sum(xValues.map((xi, i) => (xi - xMean) * (yValues[i] - yMean)));
  const denominator = Math.sqrt(d3.sum(xValues.map(xi => Math.pow(xi - xMean, 2))) * d3.sum(yValues.map(yi => Math.pow(yi - yMean, 2))));
  return numerator / denominator;
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

// ! Scatter Plot
function plotCorrelogram(data, correlationMatrix) {
  return; // ! Disable correlogram for now 
  // Extract unique variables from the data
  var allVar = [...new Set(correlationMatrix.map(d => d.group))];
  var numVar = allVar.length;

  // Dimension of the whole chart. Only one size since it has to be square
  var marginWhole = {top: 10, right: 10, bottom: 10, left: 10},
      sizeWhole = 640 - marginWhole.left - marginWhole.right;

  // Create the svg area
  var svg = d3.select("#canvas-2")
    .append("svg")
      .attr("width", sizeWhole + marginWhole.left + marginWhole.right)
      .attr("height", sizeWhole + marginWhole.top + marginWhole.bottom)
    .append("g")
      .attr("transform", "translate(" + marginWhole.left + "," + marginWhole.top + ")");

  // Now I can compute the size of a single chart
  var mar = 20;
  var size = sizeWhole / numVar;

  // Create a scale: gives the position of each pair each variable
  var position = d3.scalePoint()
    .domain(allVar)
    .range([0, sizeWhole - size]);

  // Color scale: give me a value, I return a color
  var color = d3.scaleSequential()
    .interpolator(d3.interpolateGnBu)
    .domain([-1, 1]);

  // Add charts
  for (var i in allVar) {
    for (var j in allVar) {
      // Get current variable name
      var var1 = allVar[i];
      var var2 = allVar[j];

      // If var1 == var2 I'm on the diagonal, I skip that
      if (var1 === var2) { continue; }

      // Filter data for the current pair of variables
      var filteredData = data.map(d => ({ x: d[var1], y: d[var2] }));

      // Add X Scale of each graph
      var xextent = d3.extent(filteredData, function(d) { return +d.x; });
      var x = d3.scaleLinear()
        .domain(xextent).nice()
        .range([0, size - 2 * mar]);

      // Add Y Scale of each graph
      var yextent = d3.extent(filteredData, function(d) { return +d.y; });
      var y = d3.scaleLinear()
        .domain(yextent).nice()
        .range([size - 2 * mar, 0]);

      // Add a 'g' at the right position
      var tmp = svg
        .append('g')
        .attr("transform", "translate(" + (position(var1) + mar) + "," + (position(var2) + mar) + ")");

      // Add X and Y axis in tmp
      tmp.append("g")
        .attr("transform", "translate(0," + (size - mar * 2) + ")")
        .call(d3.axisBottom(x).ticks(3));
      tmp.append("g")
        .call(d3.axisLeft(y).ticks(3));

      // Add circles
      tmp.selectAll("myCircles")
        .data(filteredData)
        .enter()
        .append("circle")
          .attr("cx", function(d) { return x(+d.x); })
          .attr("cy", function(d) { return y(+d.y); })
          .attr("r", 3)
          .attr("fill", function(d) { return color(d3.mean([d.x, d.y])); });
    }
  }

  // Add variable names = diagonal
  for (var i in allVar) {
    for (var j in allVar) {
      // If var1 == var2 I'm on the diagonal, otherwise I skip
      if (i != j) { continue; }
      // Add text
      var var1 = allVar[i];
      var var2 = allVar[j];
      svg.append('g')
        .attr("transform", "translate(" + position(var1) + "," + position(var2) + ")")
        .append('text')
          .attr("x", size / 2)
          .attr("y", size / 2)
          .text(var1)
          .attr("text-anchor", "middle");
    }
  }
}
