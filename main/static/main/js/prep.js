document.addEventListener("DOMContentLoaded", function() {
  showSection("missing_value_section");

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

document.getElementById('file').addEventListener('change',preview_data);

const featureSelectionDiv = document.getElementById("feature_selection");
const encoding_selection = document.getElementById("encoding_selection");
const scale_selection = document.getElementById("scale_selection");
const feat = "feature_selection";
const enco = "encoding_selection";
const scale = "scaling_selection";

function preview_data(e) {
  const file = e.target.files[0];
  const formData = new FormData();
  formData.append('file', file);
    
  let text, rows, headers, null_columns;

  fetch('/preprocessing', {
    method: 'POST',
    body: formData,
    headers: {
      'X-CSRFToken': getCookie('csrftoken'), // Use a helper function to get CSRF token
    },
  })
  .then(response => {
    return response.json();
  })
  .then(data => {
    if (data.error) {
      alert(data.error);
    } else {
      text = JSON.parse(data.json_data);
      headers = data.headers;
      null_columns = data.null_columns;
      non_numerical_cols = data.non_numerical_cols;
      generatecolumns(null_columns, featureSelectionDiv, feat); // generate columns for missing values     
      generatecolumns(non_numerical_cols, encoding_selection, enco); // generate columns for encoding
      generatecolumns(headers, scale_selection, scale); // generate columns for scaling
      generateTable(text, null_columns); // Append the generated table to the container
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

// Function to generate table from JSON data
function generateTable(jsonData, null_columns) {
  const container = document.getElementById('csv-preview-container');

  // Clear existing content in the container
  container.innerHTML = '';

  const table = document.createElement('table');
  table.className = 'table table-bordered table-hover table-striped';
  const headerRow = document.createElement('tr');

  // Create table headers
  Object.keys(jsonData[0]).forEach(key => {
    const th = document.createElement('th');
    th.innerText = key.charAt(0).toUpperCase() + key.slice(1); // Capitalize first letter
    // Add the bg-warning class if the key is in null_columns
    if (null_columns.includes(key)) {
      th.classList.add('bg-warning');
    }
    headerRow.appendChild(th);
  });

  table.appendChild(headerRow);

  // Create table rows
  jsonData.forEach(item => {
    const row = document.createElement('tr');
    Object.values(item).forEach(value => {
      const td = document.createElement('td');
      td.innerText = value;
      row.appendChild(td);
    });
    table.appendChild(row);
  });
  
  // Append the new table to the container
  container.appendChild(table);
  //show the info button
  $('#data-info').removeClass('d-none');
}      
        
// Populate the column for selection 
function generatecolumns(columns,SelectionDiv,selection) {
  // Clear any existing checkboxes
  SelectionDiv.innerHTML = '';

  // Iterate over the columns array
  columns.forEach(column => {
    // Create a new checkbox input element
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = column + "_" + selection; // Set the checkbox id to the column name
    checkbox.value = column; // Set the checkbox value to the column name
    checkbox.className = "form-check-input"; // Add Bootstrap class for styling
    checkbox.name = selection; //feature selection or encoding selection

    // Create a label for the checkbox
    const label = document.createElement("label");
    label.htmlFor = column + "_" + selection; // Link the label to the checkbox
    label.textContent = column; // Set the label text to the column name
    label.className = "form-check-label"; // Add Bootstrap class for styling

    // Create a div to contain the checkbox and label
    const checkboxDiv = document.createElement("div");
    checkboxDiv.className = "form-check"; // Add Bootstrap class for styling
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
      alert(data.error);
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
    console.error('Error:', error);
  });
}

// Call this function when the user applies the missing value strategy
document.getElementById('missing_value_strategybtn').addEventListener('click', applyMissingValueStrategy);

//Encoding strategy
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
      alert(data.error);
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
    console.error('Error:', error);
  });
}

// Call this function when the user applies the encoding strategy
document.getElementById('encoding_strategybtn').addEventListener('click', encoding);

//scaling strategy
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
    alert(data.error);
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
    console.error('Error:', error);
});
}

// Call this function when the user applies the scaling strategy
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
    <div class="text-left">
      <h3>Missing Value Techniques</h3>
      <p>When dealing with missing values, consider the following techniques:</p>
      <ul>
        <li><strong>Remove Rows:</strong> Delete rows with missing values if they are few and won't bias your analysis.</li>
        <li><strong>Mean/Median Imputation:</strong> Replace missing values with the mean or median of the column.</li>
        <li><strong>Most Frequent:</strong> Replace missing values with the most frequent value of the column. It will work for categorical data as well.</li>
      </ul>
      
      <h3>Encoding Strategies</h3>
      <p>Choose an encoding method based on the type of data:</p>
      <ul>
        <li><strong>Label Encoding:</strong> Assign a unique integer to each category (suitable for ordinal data).</li>
        <li><strong>One-Hot Encoding:</strong> Create binary columns for each category (suitable for nominal data).</li>
      </ul>
  
      <h3>Normalization Methods</h3>
      <p>Normalize your data to ensure consistent scaling:</p>
      <ul>
        <li><strong>Normalization: </strong>Scale features to a range of [0, 1]. Useful for algorithms sensitive to scales.</li>
        <li><strong>Standardization: </strong>Transforms the data to have a mean of 0 and a standard deviation of 1.</li>
      </ul>
    </div>
    `,
    showCloseButton: true,
    focusConfirm: false,
    icon: 'info',
    confirmButtonText: 'Got it!',
    customClass: {
      popup: 'custom-popup', // Adding a custom class for styling
      htmlContainer: 'custom-html' // If you want to style just the HTML content specifically
    },
    width: '80%', // Adjust width of the alert box (increase the size)
  });
  
}


function toggleInfo() {
  Swal.fire({
    html: `<div id="data-table-container"></div>`,
    showCloseButton: true,
    focusConfirm: false,
    icon: 'info',
    confirmButtonText: 'Got it!',
    customClass: {
      popup: 'custom-popup', // Adding a custom class for styling
      htmlContainer: 'custom-html' // If you want to style just the HTML content specifically
    },
    width: '80%', // Adjust width of the alert box (increase the size)
  });

  fetch('preprocessing/scaling/data_details/', {
    // method: 'POST',
    // headers: {
    //   'Content-Type': 'application/json',
    //   'X-CSRFToken': getCookie('csrftoken'),  // Use a helper function to get CSRF token
    // },
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);
    console.log("Helloooooooooooo");
  })
  .catch(error => {
    console.error('Error:', error);
  });
  
}

function generateInfoTable(jsonData) {
  const container = document.getElementById('data-table-container');

  // Clear existing content in the container
  container.innerHTML = '';

  const table = document.createElement('table');
  table.className = 'table table-bordered table-hover table-striped';
  const headerRow = document.createElement('tr');

  // Create table headers
  Object.keys(jsonData[0]).forEach(key => {
    const th = document.createElement('th');
    th.innerText = key.charAt(0).toUpperCase() + key.slice(1); // Capitalize first letter
    // Add the bg-warning class if the key is in null_columns
    if (null_columns.includes(key)) {
      th.classList.add('bg-warning');
    }
    headerRow.appendChild(th);
  });

  table.appendChild(headerRow);

  // Create table rows
  jsonData.forEach(item => {
    const row = document.createElement('tr');
    Object.values(item).forEach(value => {
      const td = document.createElement('td');
      td.innerText = value;
      row.appendChild(td);
    });
    table.appendChild(row);
  });
  
  // Append the new table to the container
  container.appendChild(table);
  
}  