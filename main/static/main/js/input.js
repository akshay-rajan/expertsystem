function handleFileChange(event) {
  const file = event.target.files[0];
  if (file) $('#upload-btn').prop('disabled', false);
}
function handleFileUpload(event) {
  event.preventDefault();
  const fileInput = document.getElementById('dataset');
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const content = e.target.result;
      const columns = extractColumns(content, file.name);
      // Populate checkboxes
      populateFeatureCheckboxes(columns);
      // Populate target dropdown
      populateTargetDropdown(columns);
    };
    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else if (file.name.endsWith('.xls') || file.name.endsWith('.xlsx')) {
      reader.readAsBinaryString(file);
    }
    plot();
    $('#upload-btn').addClass('d-none');
    $('#build-btn').removeClass('d-none');
  }
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
    option.value = column;
    option.textContent = column;
    targetSelect.appendChild(option);
  });
}
function validateForm() {
  const features = document.querySelectorAll('input[name="features"]:checked');
  const target = document.getElementById('target').value;
  const alert = $('#alert');
  if (features.length === 0) {
    alert.text('Please select at least one feature.');
    alert.removeClass('d-none');
    return false;
  }
  if (!target) {
    alert.text('Please select a target.');
    alert.removeClass('d-none');
    return false;
  }
  return true;
}

function plot() {
  return;
  $("#canvas-1").removeClass("d-none");
  const xValues = [50,60,70,80,90,100,110,120,130,140,150];
  const yValues = [7,8,8,9,9,9,10,11,14,14,15];

  new Chart("myChart", {
    type: "line",
    data: {
      labels: xValues,
      datasets: [{
        backgroundColor:"rgba(0,0,255,1.0)",
        borderColor: "rgba(0,0,255,0.1)",
        data: yValues
      }]
    },
    options: {
      legend: {display: false},
      scales: {
        yAxes: [
          {ticks: 
            {
              min: 6, 
              max:16,
            }
          }
        ],
      }
    }
  });
}
