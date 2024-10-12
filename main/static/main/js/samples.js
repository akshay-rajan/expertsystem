// ! Generate the dataset download links on the samples page.
document.addEventListener("DOMContentLoaded", function() {
  const datasets = [
    {
      "name": "California Housing",
      "file": "fetch_california_housing.xlsx",
      "type": "XLSX",
      "for": "Regression"
    },
    {
      "name": "California Housing",
      "file": "fetch_california_housing.csv",
      "type": "CSV",
      "for": "Regression"
    },
    {
      "name": "Numerical Data",
      "file": "numerical_data.xlsx",
      "type": "XLSX",
      "for": "Regression"
    },
    {
      "name": "Iris",
      "file": "iris.csv",
      "type": "CSV",
      "for": "Classification"
    },
    {
      "name": "Mall Customers",
      "file": "mall_customers.csv",
      "type": "CSV",
      "for": "Clustering"
    }
  ];

  const container = document.getElementById("dataset-container");

  datasets.forEach(dataset => {
    const card = document.createElement("div");
    card.className = "card m-2 p-2";
    card.style.minWidth = "18rem";

    const cardBody = document.createElement("div");
    cardBody.className = "card-body";

    const title = document.createElement("div");
    title.textContent = dataset.name;
    title.className = "font-weight-bold card-title h5";
    // const link = document.createElement("a");
    // link.href = `/static/main/files/${dataset.file}`;
    // link.download = dataset.name;
    // link.textContent = dataset.name;
    // link.className = "card-title h5";

    const cardContent = document.createElement("div");
    cardContent.className = "d-flex justify-content-between";

    const info = document.createElement("div");
    
    const type = document.createElement("div");
    type.className = "font-weight-bold text-muted";
    type.textContent = `Type: ${dataset.type}`;

    const forInfo = document.createElement("div");
    forInfo.className = "text-muted";
    forInfo.textContent = `For ${dataset.for}`;

    const download = document.createElement("a");
    download.className = "btn";
    download.href = `/static/main/files/${dataset.file}`;
    download.download = dataset.file;
    download.innerHTML = '<img src="/static/main/img/download.svg" alt="Get" width="30" height="30">';
    
    cardBody.appendChild(title);
    cardBody.appendChild(cardContent);
    cardContent.appendChild(info);
    cardContent.appendChild(download);

    info.appendChild(type);
    info.appendChild(forInfo);
    card.appendChild(cardBody);
    container.appendChild(card);
  });
});