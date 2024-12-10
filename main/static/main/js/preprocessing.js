document.addEventListener("DOMContentLoaded", function() {
  // Enable Bootstrap popovers
  const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]')
  const popoverList = [...popoverTriggerList].map(popoverTriggerEl => {
    const popover = new bootstrap.Popover(popoverTriggerEl);

    // Show popover on mouseover
    popoverTriggerEl.addEventListener('mouseover', () => {
      popover.show();
    });

    // Hide popover on mouseout
    popoverTriggerEl.addEventListener('mouseout', () => {
      popover.hide();
    });

    return popover;
  });  
});

function handleChange(event) {
  const file = event.target.files[0];
  if (file) $('#upload-btn').prop('disabled', false);
}

function handlePreloadedFileChange(event) {
  const dataset = event.target.value;
  if (dataset) $('#upload-btn').prop('disabled', false);
}

function initiatePreprocessing(event) {
  $('.preprocessing-dataset-selection').addClass('d-none');

  const isUpload = document.getElementById('option-upload').checked;
  if (isUpload) {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    fileInput.parentElement.innerHTML = file.name + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
    preview_data(formData);
  } else {
    const dataset = document.getElementById('preloaded-dataset').value;
    if (!dataset) {
      showWarningToast('Please select a preloaded dataset.');
      return;
    }
    const preloadedDiv = document.getElementById('preloaded-div');
    preloadedDiv.innerHTML = dataset + '<img src="/static/main/img/tick.svg" class="d-inline ml-2 icon tick" alt="tick">';
    const formData = new FormData();
    formData.append('preloaded_dataset', dataset);
    preview_data(formData);
  }

  $('#upload-btn').addClass('d-none');
  $('.sections').removeClass('d-none');
  $('.sections-btn').removeClass('d-none');
}

const featureSelectionDiv = document.getElementById("feature_selection");
const encoding_selection = document.getElementById("encoding_selection");
const scale_selection = document.getElementById("scale_selection");
const feat = "feature_selection";
const enco = "encoding_selection";
const scale = "scaling_selection";

function preview_data(formData) {    
  let text, headers, null_columns;

  fetch('/preprocessing', {
    method: 'POST',
    body: formData,
    headers: {
      'X-CSRFToken': getCSRFToken(),
    },
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showError('Error!', data.error);
    } else {
      text = JSON.parse(data.json_data);
      headers = data.headers;
      null_columns = data.null_columns;
      non_numerical_cols = data.non_numerical_cols;
      generatecolumns(null_columns, featureSelectionDiv, "feature_selection"); // Generate columns for missing values     
      generatecolumns(non_numerical_cols, encoding_selection, "encoding_selection"); // Generate columns for encoding
      generatecolumns(headers, scale_selection, "scaling_selection"); // Generate columns for scaling
      generateTable(text, null_columns); // Append the generated table to the container
    }
  })
  .catch(error => {
    showError('Error!', 'An error occurred. Please try again.');
  });
}

// Function to generate table from JSON data
function generateTable(jsonData, null_columns) {
  const container = document.getElementById('csv-preview-container');

  // Clear existing content
  container.innerHTML = '';

  // Create table
  const table = document.createElement('table');
  table.className = 'table table-bordered table-hover table-striped';

  // Create table headers
  const headerRow = document.createElement('tr');
  Object.keys(jsonData[0]).forEach(key => {
    const th = document.createElement('th');
    th.innerText = key.charAt(0).toUpperCase() + key.slice(1);
    if (null_columns.includes(key)) {
      th.classList.add('bg-warning'); // Highlight null columns
    }
    headerRow.appendChild(th);
  });

  const thead = document.createElement('thead');
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Create table body
  const tbody = document.createElement('tbody');
  jsonData.forEach(item => {
    const row = document.createElement('tr');
    Object.values(item).forEach(value => {
      const td = document.createElement('td');
      td.innerText = value;
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
  table.appendChild(tbody);

  // Append table to container
  container.appendChild(table);
  $('#data-info').removeClass('d-none');
}
        
// Populate the column for selection 
function generatecolumns(columns, SelectionDiv, selection) {
  // Clear any existing checkboxes
  SelectionDiv.innerHTML = '';

  // Iterate over the columns array
  columns.forEach(column => {
    // Create a new checkbox input element
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = column + "_" + selection; // Set the checkbox id to the column name
    checkbox.value = column; // Set the checkbox value to the column name
    checkbox.className = "form-check-input"; // Bootstrap 5 class for input
    checkbox.name = selection; // Feature selection or encoding selection

    // Create a label for the checkbox
    const label = document.createElement("label");
    label.htmlFor = column + "_" + selection; // Link the label to the checkbox
    label.textContent = column; // Set the label text to the column name
    label.className = "form-check-label"; // Bootstrap 5 class for label

    // Create a div to contain the checkbox and label
    const checkboxDiv = document.createElement("div");
    checkboxDiv.className = "form-check"; // Bootstrap 5 class for grouping
    checkboxDiv.appendChild(checkbox);
    checkboxDiv.appendChild(label);

    // Append the checkbox div to the feature selection div
    SelectionDiv.appendChild(checkboxDiv);
  });
}
 
// Get the CSRF token from the cookie
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

function applyMissingValueStrategy() {
  const strategy = document.getElementById("missing_value_strategy").value;
  const selectedColumns = Array.from(document.querySelectorAll('input[name="feature_selection"]:checked')).map(checkbox => checkbox.value);

  fetch('/preprocessing/fill-missing/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken'),  // Use a helper function to get CSRF token
    },
    body: JSON.stringify({
      strategy: strategy,
      columns: selectedColumns
    }),
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showWarningToast(data.error);
    } else {
      text=JSON.parse(data.json_data)        
      headers=data.headers
      null_columns=data.null_columns
      non_numerical_cols=data.non_numerical_cols
      generatecolumns(null_columns,featureSelectionDiv,feat)    //generate columns for missing values     
      generatecolumns(non_numerical_cols,encoding_selection,enco)         //generate columns for encoding
      generatecolumns(headers,scale_selection,scale)
      generateTable(text,null_columns); // Append the generated table to the container
    }
  })
  .catch(error => {
    showWarningToast(`An error occurred: ${error}`);
  });
}
document.getElementById('missing_value_strategybtn').addEventListener('click', applyMissingValueStrategy);


function encoding() {
  const strategy = document.getElementById("encoding_strategy").value;
  const selectedColumns = Array.from(document.querySelectorAll('input[name="encoding_selection"]:checked'))
      .map(input => input.value);

  fetch('/preprocessing/encoding/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken'),  // Use a helper function to get CSRF token
    },
    body: JSON.stringify({
      strategy: strategy,
      columns: selectedColumns
    }),
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showWarningToast(data.error);
    } else {
        
      text=JSON.parse(data.json_data)        
      headers=data.headers
      null_columns=data.null_columns
      non_numerical_cols=data.non_numerical_cols
      generatecolumns(null_columns,featureSelectionDiv,feat)    //generate columns for missing values     
      generatecolumns(non_numerical_cols,encoding_selection,enco)         //generate columns for encoding
      generatecolumns(headers,scale_selection,scale)
      generateTable(text,null_columns); // Append the generated table to the container
    }
  })
  .catch(error => { 
    showWarningToast(`An error occurred: ${error}`);
  });
}
document.getElementById('encoding_strategybtn').addEventListener('click', encoding);


function applyScalingStrategy() {
  const scalingStrategy = document.getElementById("scaling_strategy").value;
  const selectedColumns = Array.from(document.querySelectorAll('input[name="scaling_selection"]:checked'))
    .map(input => input.value);

  fetch('/preprocessing/scaling/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken'), 
    },
    body: JSON.stringify({
      strategy: scalingStrategy,
      columns: selectedColumns
    }),
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showWarningToast(data.error);
    } else {
      text=JSON.parse(data.json_data)        
      headers=data.headers
      null_columns=data.null_columns
      non_numerical_cols=data.non_numerical_cols
      generatecolumns(null_columns,featureSelectionDiv,feat)    //generate columns for missing values     
      generatecolumns(non_numerical_cols,encoding_selection,enco)         //generate columns for encoding
      generatecolumns(headers,scale_selection,scale)
      generateTable(text,null_columns); // Append the generated table to the container
    }
  })
  .catch(error => {
    showWarningToast(`An error occurred: ${error}`);
  });
}
document.getElementById('scaling_strategybtn').addEventListener('click', applyScalingStrategy);

// Function to control the visibility of sections
function showSection(sectionId) {
  // Hide all sections
  document.getElementById("missing_value_section").style.display = "none";
  document.getElementById("encoding_section").style.display = "none";
  document.getElementById("scaling_section").style.display = "none";
  // Reset class of all buttons
  document.getElementById("missing_value_section-btn").classList = "btn btn-primary mb-auto w-100";
  document.getElementById("encoding_section-btn").classList = "btn btn-primary mb-auto w-100";
  document.getElementById("scaling_section-btn").classList = "btn btn-primary mb-auto w-100";

  // Mark the selected section
  let thisBtn = document.getElementById(sectionId + "-btn");
  thisBtn.classList.remove("btn-primary");
  thisBtn.classList.add("btn-outline-primary");
  thisBtn.classList.add("disabled");

  // Show the selected section
  document.getElementById(sectionId).style.display = "block";
}

function toggleGuide() {
  Swal.fire({
    html: `
    <div class="text-start">
      <h3 class="mb-3">Missing Value Techniques</h3>
      <p>When dealing with missing values, consider the following techniques:</p>
      <ul>
        <li><strong>Remove Rows:</strong> Delete rows with missing values if they are few and won't bias your analysis.</li>
        <li><strong>Mean/Median Imputation:</strong> Replace missing values with the mean or median of the column.</li>
        <li><strong>Most Frequent:</strong> Replace missing values with the most frequent value of the column. It will work for categorical data as well.</li>
      </ul>
      
      <h3 class="mt-4 mb-3">Encoding Strategies</h3>
      <p>Choose an encoding method based on the type of data:</p>
      <ul>
        <li><strong>Label Encoding:</strong> Assign a unique integer to each category (suitable for ordinal data).</li>
        <li><strong>One-Hot Encoding:</strong> Create binary columns for each category (suitable for nominal data).</li>
      </ul>
  
      <h3 class="mt-4 mb-3">Normalization Methods</h3>
      <p>Normalize your data to ensure consistent scaling:</p>
      <ul>
        <li><strong>Normalization: </strong>Scale features to a range of [0, 1]. Useful for algorithms sensitive to scales.</li>
        <li><strong>Standardization: </strong>Transforms the data to have a mean of 0 and a standard deviation of 1.</li>
      </ul>
    </div>
    `,
    showCloseButton: true,
    focusConfirm: false,
    showConfirmButton: false,
    customClass: {
      popup: 'custom-popup', 
      htmlContainer: 'custom-html'
    },
    width: '80%',
  });
}


// Get data details and display in a table inside a SweetAlert modal
function toggleInfo() {
  Swal.fire({
    title: 'Data Details',
    html: `<div id="data-table-container"></div>`, // The div where the table will be inserted
    showCloseButton: true,
    focusConfirm: false,
    confirmButtonText: 'Close',
    customClass: {
      popup: 'custom-popup', // Custom class for the popup
      htmlContainer: 'table-info-container' // Custom class for HTML content inside popup
    },
    width: '80%', // Adjust width of the alert box (increase the size)
  });

  fetch('preprocessing/scaling/data_details/', {
    // method: 'POST',  // Uncomment if needed for your request
    // headers: {
    //   'Content-Type': 'application/json',
    //   'X-CSRFToken': getCookie('csrftoken'),  // Add CSRF token if required
    // },
  })
  .then(response => response.json())
  .then(data => {
    // Build the table for column-wise details
    let tableHtml = '<table class="table table-bordered table-striped table-hover">'; // Add `table-hover` for better UX
    tableHtml += '<thead class="table-light"><tr><th>Column Name</th><th>Data Type</th><th>Non-null Count</th><th>Missing Values</th></tr></thead>';
    tableHtml += '<tbody>';

    // Loop through columns in the data and create table rows
    for (let col in data.data_types) {
      tableHtml += `
        <tr>
          <td>${col}</td>
          <td>${data.data_types[col]}</td>
          <td>${data.non_null_counts[col]}</td>
          <td>${data.missing_values[col]}</td>
        </tr>
      `;
    }

    tableHtml += '</tbody>';
    tableHtml += '</table>';

    // Render the column table inside the Swal popup
    document.getElementById('data-table-container').innerHTML = tableHtml;

    // Append additional information (Shape, Memory Usage, Numeric Summary, and Data Info)
    let additionalInfoHtml = `
      <h5 class="mt-4">Dataset Shape</h5>
      <ul class="list-unstyled">
        <li><strong>Rows:</strong> ${data.shape[0]}</li>
        <li><strong>Columns:</strong> ${data.shape[1]}</li>
      </ul>
      
      <h5 class="mt-4">Memory Usage</h5>
      <table class="table table-bordered table-striped table-hover">
        <thead class="table-light">
          <tr><th>Column Name</th><th>Memory Usage</th></tr>
        </thead>
        <tbody>
          ${Object.keys(data.memory_usage).map(col => {
            return `
              <tr>
                <td>${col}</td>
                <td>${data.memory_usage[col]}</td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
      
      <h5 class="mt-4">Numeric Summary</h5>
      <table class="table table-bordered table-striped table-hover">
        <thead class="table-light">
          <tr>
            <th>Column Name</th>
            <th>Count</th>
            <th>Mean</th>
            <th>Min</th>
            <th>Max</th>
            <th>Standard Deviation</th>
          </tr>
        </thead>
        <tbody>
          ${Object.keys(data.numeric_summary).map(col => {
            return `
              <tr>
                <td>${col}</td>
                <td>${data.numeric_summary[col].count}</td>
                <td>${data.numeric_summary[col].mean}</td>
                <td>${data.numeric_summary[col].min}</td>
                <td>${data.numeric_summary[col].max}</td>
                <td>${data.numeric_summary[col].std}</td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
      
      <h5 class="mt-4">Data Info</h5>
      <div class="data-info-box">
        <pre class="bg-light p-3 border rounded">${data.data_info}</pre> <!-- Use Bootstrap 5 utilities -->
      </div>
    `;

    // Append the additional information below the table
    document.getElementById('data-table-container').innerHTML += additionalInfoHtml;

  })
  .catch(error => {
    showWarningToast(`An error occurred: ${error}`);
  });
}

function toggleCategories() {
  document.getElementById('category-display').classList.toggle('d-none');
  document.getElementById('training-btn-div').classList.toggle('d-none');
}

